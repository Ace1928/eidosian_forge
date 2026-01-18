from pyparsing import (
import pydot
def push_edge_stmt(s, loc, toks):
    tok_attrs = [a for a in toks if isinstance(a, P_AttrList)]
    attrs = {}
    for a in tok_attrs:
        attrs.update(a.attrs)
    e = []
    if isinstance(toks[0][0], pydot.Graph):
        n_prev = pydot.frozendict(toks[0][0].obj_dict)
    else:
        n_prev = toks[0][0] + do_node_ports(toks[0])
    if isinstance(toks[2][0], ParseResults):
        n_next_list = [[n.get_name()] for n in toks[2][0]]
        for n_next in [n for n in n_next_list]:
            n_next_port = do_node_ports(n_next)
            e.append(pydot.Edge(n_prev, n_next[0] + n_next_port, **attrs))
    elif isinstance(toks[2][0], pydot.Graph):
        e.append(pydot.Edge(n_prev, pydot.frozendict(toks[2][0].obj_dict), **attrs))
    elif isinstance(toks[2][0], pydot.Node):
        node = toks[2][0]
        if node.get_port() is not None:
            name_port = node.get_name() + ':' + node.get_port()
        else:
            name_port = node.get_name()
        e.append(pydot.Edge(n_prev, name_port, **attrs))
    elif isinstance(toks[2][0], str):
        for n_next in [n for n in tuple(toks)[2::2]]:
            if isinstance(n_next, P_AttrList) or not isinstance(n_next[0], str):
                continue
            n_next_port = do_node_ports(n_next)
            e.append(pydot.Edge(n_prev, n_next[0] + n_next_port, **attrs))
            n_prev = n_next[0] + n_next_port
    else:
        raise Exception('Edge target {r} with type {s} unsupported.'.format(r=toks[2][0], s=type(toks[2][0])))
    return e