import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def parse_gml_lines(lines, label, destringizer):
    """Parse GML `lines` into a graph."""

    def tokenize():
        patterns = ['[A-Za-z][0-9A-Za-z_]*\\b', '[+-]?(?:[0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*|INF)(?:[Ee][+-]?[0-9]+)?', '[+-]?[0-9]+', '".*?"', '\\[', '\\]', '#.*$|\\s+']
        tokens = re.compile('|'.join((f'({pattern})' for pattern in patterns)))
        lineno = 0
        multilines = []
        for line in lines:
            pos = 0
            if multilines:
                multilines.append(line.strip())
                if line[-1] == '"':
                    line = ' '.join(multilines)
                    multilines = []
                else:
                    lineno += 1
                    continue
            elif line.count('"') == 1:
                if line.strip()[0] != '"' and line.strip()[-1] != '"':
                    multilines = [line.rstrip()]
                    lineno += 1
                    continue
            length = len(line)
            while pos < length:
                match = tokens.match(line, pos)
                if match is None:
                    m = f'cannot tokenize {line[pos:]} at ({lineno + 1}, {pos + 1})'
                    raise NetworkXError(m)
                for i in range(len(patterns)):
                    group = match.group(i + 1)
                    if group is not None:
                        if i == 0:
                            value = group.rstrip()
                        elif i == 1:
                            value = float(group)
                        elif i == 2:
                            value = int(group)
                        else:
                            value = group
                        if i != 6:
                            yield Token(Pattern(i), value, lineno + 1, pos + 1)
                        pos += len(group)
                        break
            lineno += 1
        yield Token(None, None, lineno + 1, 1)

    def unexpected(curr_token, expected):
        category, value, lineno, pos = curr_token
        value = repr(value) if value is not None else 'EOF'
        raise NetworkXError(f'expected {expected}, found {value} at ({lineno}, {pos})')

    def consume(curr_token, category, expected):
        if curr_token.category == category:
            return next(tokens)
        unexpected(curr_token, expected)

    def parse_kv(curr_token):
        dct = defaultdict(list)
        while curr_token.category == Pattern.KEYS:
            key = curr_token.value
            curr_token = next(tokens)
            category = curr_token.category
            if category == Pattern.REALS or category == Pattern.INTS:
                value = curr_token.value
                curr_token = next(tokens)
            elif category == Pattern.STRINGS:
                value = unescape(curr_token.value[1:-1])
                if destringizer:
                    try:
                        value = destringizer(value)
                    except ValueError:
                        pass
                if value == '()':
                    value = ()
                if value == '[]':
                    value = []
                curr_token = next(tokens)
            elif category == Pattern.DICT_START:
                curr_token, value = parse_dict(curr_token)
            elif key in ('id', 'label', 'source', 'target'):
                try:
                    value = unescape(str(curr_token.value))
                    if destringizer:
                        try:
                            value = destringizer(value)
                        except ValueError:
                            pass
                    curr_token = next(tokens)
                except Exception:
                    msg = "an int, float, string, '[' or string" + ' convertible ASCII value for node id or label'
                    unexpected(curr_token, msg)
            elif curr_token.value in {'NAN', 'INF'}:
                value = float(curr_token.value)
                curr_token = next(tokens)
            else:
                unexpected(curr_token, "an int, float, string or '['")
            dct[key].append(value)

        def clean_dict_value(value):
            if not isinstance(value, list):
                return value
            if len(value) == 1:
                return value[0]
            if value[0] == LIST_START_VALUE:
                return value[1:]
            return value
        dct = {key: clean_dict_value(value) for key, value in dct.items()}
        return (curr_token, dct)

    def parse_dict(curr_token):
        curr_token = consume(curr_token, Pattern.DICT_START, "'['")
        curr_token, dct = parse_kv(curr_token)
        curr_token = consume(curr_token, Pattern.DICT_END, "']'")
        return (curr_token, dct)

    def parse_graph():
        curr_token, dct = parse_kv(next(tokens))
        if curr_token.category is not None:
            unexpected(curr_token, 'EOF')
        if 'graph' not in dct:
            raise NetworkXError('input contains no graph')
        graph = dct['graph']
        if isinstance(graph, list):
            raise NetworkXError('input contains more than one graph')
        return graph
    tokens = tokenize()
    graph = parse_graph()
    directed = graph.pop('directed', False)
    multigraph = graph.pop('multigraph', False)
    if not multigraph:
        G = nx.DiGraph() if directed else nx.Graph()
    else:
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    graph_attr = {k: v for k, v in graph.items() if k not in ('node', 'edge')}
    G.graph.update(graph_attr)

    def pop_attr(dct, category, attr, i):
        try:
            return dct.pop(attr)
        except KeyError as err:
            raise NetworkXError(f'{category} #{i} has no {attr!r} attribute') from err
    nodes = graph.get('node', [])
    mapping = {}
    node_labels = set()
    for i, node in enumerate(nodes if isinstance(nodes, list) else [nodes]):
        id = pop_attr(node, 'node', 'id', i)
        if id in G:
            raise NetworkXError(f'node id {id!r} is duplicated')
        if label is not None and label != 'id':
            node_label = pop_attr(node, 'node', label, i)
            if node_label in node_labels:
                raise NetworkXError(f'node label {node_label!r} is duplicated')
            node_labels.add(node_label)
            mapping[id] = node_label
        G.add_node(id, **node)
    edges = graph.get('edge', [])
    for i, edge in enumerate(edges if isinstance(edges, list) else [edges]):
        source = pop_attr(edge, 'edge', 'source', i)
        target = pop_attr(edge, 'edge', 'target', i)
        if source not in G:
            raise NetworkXError(f'edge #{i} has undefined source {source!r}')
        if target not in G:
            raise NetworkXError(f'edge #{i} has undefined target {target!r}')
        if not multigraph:
            if not G.has_edge(source, target):
                G.add_edge(source, target, **edge)
            else:
                arrow = '->' if directed else '--'
                msg = f'edge #{i} ({source!r}{arrow}{target!r}) is duplicated'
                raise nx.NetworkXError(msg)
        else:
            key = edge.pop('key', None)
            if key is not None and G.has_edge(source, target, key):
                arrow = '->' if directed else '--'
                msg = f'edge #{i} ({source!r}{arrow}{target!r}, {key!r})'
                msg2 = 'Hint: If multigraph add "multigraph 1" to file header.'
                raise nx.NetworkXError(msg + ' is duplicated\n' + msg2)
            G.add_edge(source, target, key, **edge)
    if label is not None and label != 'id':
        G = nx.relabel_nodes(G, mapping)
    return G