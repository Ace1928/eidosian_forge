import networkx as nx
from networkx.utils.decorators import not_implemented_for
@nx._dispatch(graphs={'t1': 0, 't2': 2})
def root_trees(t1, root1, t2, root2):
    """Create a single digraph dT of free trees t1 and t2
    #   with roots root1 and root2 respectively
    # rename the nodes with consecutive integers
    # so that all nodes get a unique name between both trees

    # our new "fake" root node is 0
    # t1 is numbers from 1 ... n
    # t2 is numbered from n+1 to 2n
    """
    dT = nx.DiGraph()
    newroot1 = 1
    newroot2 = nx.number_of_nodes(t1) + 1
    namemap1 = {root1: newroot1}
    namemap2 = {root2: newroot2}
    dT.add_edge(0, namemap1[root1])
    dT.add_edge(0, namemap2[root2])
    for i, (v1, v2) in enumerate(nx.bfs_edges(t1, root1)):
        namemap1[v2] = i + namemap1[root1] + 1
        dT.add_edge(namemap1[v1], namemap1[v2])
    for i, (v1, v2) in enumerate(nx.bfs_edges(t2, root2)):
        namemap2[v2] = i + namemap2[root2] + 1
        dT.add_edge(namemap2[v1], namemap2[v2])
    namemap = {}
    for old, new in namemap1.items():
        namemap[new] = old
    for old, new in namemap2.items():
        namemap[new] = old
    return (dT, namemap, newroot1, newroot2)