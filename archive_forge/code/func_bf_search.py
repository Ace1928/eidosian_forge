from functools import reduce
def bf_search(graph, root=None):
    """Breadth first search of g.

    Returns a list of all nodes that can be reached from the root node
    in breadth-first order.

    If root is not given, the search will be rooted at an arbitrary node.
    """
    seen = {}
    search = []
    if len(graph.nodes()) < 1:
        return search
    if root is None:
        root = graph.nodes()[0]
    seen[root] = 1
    search.append(root)
    current = graph.children(root)
    while len(current) > 0:
        node = current[0]
        current = current[1:]
        if node not in seen:
            search.append(node)
            seen[node] = 1
            current.extend(graph.children(node))
    return search