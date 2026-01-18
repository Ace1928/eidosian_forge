import networkx as nx
from networkx.utils.decorators import not_implemented_for
def generate_isomorphism(v, w, M, ordered_children):
    assert v < w
    M.append((v, w))
    for i, (x, y) in enumerate(zip(ordered_children[v], ordered_children[w])):
        generate_isomorphism(x, y, M, ordered_children)