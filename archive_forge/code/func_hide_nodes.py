def hide_nodes(nodes):
    nodes = set(nodes)
    return lambda node: node not in nodes