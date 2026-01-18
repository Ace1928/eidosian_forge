def hide_multidiedges(edges):
    edges = {(u, v, k) for u, v, k in edges}
    return lambda u, v, k: (u, v, k) not in edges