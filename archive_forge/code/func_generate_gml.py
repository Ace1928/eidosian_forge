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
def generate_gml(G, stringizer=None):
    """Generate a single entry of the graph `G` in GML format.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be converted to GML.

    stringizer : callable, optional
        A `stringizer` which converts non-int/non-float/non-dict values into
        strings. If it cannot convert a value into a string, it should raise a
        `ValueError` to indicate that. Default value: None.

    Returns
    -------
    lines: generator of strings
        Lines of GML data. Newlines are not appended.

    Raises
    ------
    NetworkXError
        If `stringizer` cannot convert a value into a string, or the value to
        convert is not a string while `stringizer` is None.

    See Also
    --------
    literal_stringizer

    Notes
    -----
    Graph attributes named 'directed', 'multigraph', 'node' or
    'edge', node attributes named 'id' or 'label', edge attributes
    named 'source' or 'target' (or 'key' if `G` is a multigraph)
    are ignored because these attribute names are used to encode the graph
    structure.

    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_node("1")
    >>> print("\\n".join(nx.generate_gml(G)))
    graph [
      node [
        id 0
        label "1"
      ]
    ]
    >>> G = nx.MultiGraph([("a", "b"), ("a", "b")])
    >>> print("\\n".join(nx.generate_gml(G)))
    graph [
      multigraph 1
      node [
        id 0
        label "a"
      ]
      node [
        id 1
        label "b"
      ]
      edge [
        source 0
        target 1
        key 0
      ]
      edge [
        source 0
        target 1
        key 1
      ]
    ]
    """
    valid_keys = re.compile('^[A-Za-z][0-9A-Za-z_]*$')

    def stringize(key, value, ignored_keys, indent, in_list=False):
        if not isinstance(key, str):
            raise NetworkXError(f'{key!r} is not a string')
        if not valid_keys.match(key):
            raise NetworkXError(f'{key!r} is not a valid key')
        if not isinstance(key, str):
            key = str(key)
        if key not in ignored_keys:
            if isinstance(value, (int, bool)):
                if key == 'label':
                    yield (indent + key + ' "' + str(value) + '"')
                elif value is True:
                    yield (indent + key + ' 1')
                elif value is False:
                    yield (indent + key + ' 0')
                elif value < -2 ** 31 or value >= 2 ** 31:
                    yield (indent + key + ' "' + str(value) + '"')
                else:
                    yield (indent + key + ' ' + str(value))
            elif isinstance(value, float):
                text = repr(value).upper()
                if text == repr(float('inf')).upper():
                    text = '+' + text
                else:
                    epos = text.rfind('E')
                    if epos != -1 and text.find('.', 0, epos) == -1:
                        text = text[:epos] + '.' + text[epos:]
                if key == 'label':
                    yield (indent + key + ' "' + text + '"')
                else:
                    yield (indent + key + ' ' + text)
            elif isinstance(value, dict):
                yield (indent + key + ' [')
                next_indent = indent + '  '
                for key, value in value.items():
                    yield from stringize(key, value, (), next_indent)
                yield (indent + ']')
            elif isinstance(value, tuple) and key == 'label':
                yield (indent + key + f' "({','.join((repr(v) for v in value))})"')
            elif isinstance(value, (list, tuple)) and key != 'label' and (not in_list):
                if len(value) == 0:
                    yield (indent + key + ' ' + f'"{value!r}"')
                if len(value) == 1:
                    yield (indent + key + ' ' + f'"{LIST_START_VALUE}"')
                for val in value:
                    yield from stringize(key, val, (), indent, True)
            else:
                if stringizer:
                    try:
                        value = stringizer(value)
                    except ValueError as err:
                        raise NetworkXError(f'{value!r} cannot be converted into a string') from err
                if not isinstance(value, str):
                    raise NetworkXError(f'{value!r} is not a string')
                yield (indent + key + ' "' + escape(value) + '"')
    multigraph = G.is_multigraph()
    yield 'graph ['
    if G.is_directed():
        yield '  directed 1'
    if multigraph:
        yield '  multigraph 1'
    ignored_keys = {'directed', 'multigraph', 'node', 'edge'}
    for attr, value in G.graph.items():
        yield from stringize(attr, value, ignored_keys, '  ')
    node_id = dict(zip(G, range(len(G))))
    ignored_keys = {'id', 'label'}
    for node, attrs in G.nodes.items():
        yield '  node ['
        yield ('    id ' + str(node_id[node]))
        yield from stringize('label', node, (), '    ')
        for attr, value in attrs.items():
            yield from stringize(attr, value, ignored_keys, '    ')
        yield '  ]'
    ignored_keys = {'source', 'target'}
    kwargs = {'data': True}
    if multigraph:
        ignored_keys.add('key')
        kwargs['keys'] = True
    for e in G.edges(**kwargs):
        yield '  edge ['
        yield ('    source ' + str(node_id[e[0]]))
        yield ('    target ' + str(node_id[e[1]]))
        if multigraph:
            yield from stringize('key', e[2], (), '    ')
        for attr, value in e[-1].items():
            yield from stringize(attr, value, ignored_keys, '    ')
        yield '  ]'
    yield ']'