import networkx as nx
@nx._dispatch(edge_attrs='weight')
def flow_matrix_row(G, weight=None, dtype=float, solver='lu'):
    import numpy as np
    solvername = {'full': FullInverseLaplacian, 'lu': SuperLUInverseLaplacian, 'cg': CGInverseLaplacian}
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=range(n), weight=weight).asformat('csc')
    L = L.astype(dtype)
    C = solvername[solver](L, dtype=dtype)
    w = C.w
    for u, v in sorted((sorted((u, v)) for u, v in G.edges())):
        B = np.zeros(w, dtype=dtype)
        c = G[u][v].get(weight, 1.0)
        B[u % w] = c
        B[v % w] = -c
        row = B @ C.get_rows(u, v)
        yield (row, (u, v))