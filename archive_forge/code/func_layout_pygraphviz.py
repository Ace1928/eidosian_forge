import itertools
import functools
import importlib.util
def layout_pygraphviz(G, prog='neato', dim=2, **kwargs):
    import pygraphviz as pgv
    aG = pgv.AGraph(directed=G.is_directed())
    mapping = {}
    for nodea, nodeb in G.edges():
        s_nodea = str(nodea)
        s_nodeb = str(nodeb)
        mapping[s_nodea] = nodea
        mapping[s_nodeb] = nodeb
        aG.add_edge(s_nodea, s_nodeb)
    kwargs = {}
    if dim == 2.5:
        kwargs['dim'] = 3
        kwargs['dimen'] = 2
    else:
        kwargs['dim'] = kwargs['dimen'] = dim
    args = ' '.join((f'-G{k}={v}' for k, v in kwargs.items()))
    aG.layout(prog=prog, args=args)
    pos = {}
    for snode, node in mapping.items():
        spos = aG.get_node(snode).attr['pos']
        pos[node] = tuple(map(float, spos.split(',')))
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmaz = float('-inf')
    for x, y, *maybe_z in pos.values():
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        for z in maybe_z:
            zmin = min(zmin, z)
            zmaz = max(zmaz, z)
    for node, (x, y, *maybe_z) in pos.items():
        pos[node] = (2 * (x - xmin) / (xmax - xmin) - 1, 2 * (y - ymin) / (ymax - ymin) - 1, *(2 * (z - zmin) / (zmaz - zmin) - 1 for z in maybe_z))
    return pos