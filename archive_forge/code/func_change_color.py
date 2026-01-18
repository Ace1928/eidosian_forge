from collections import defaultdict
import networkx as nx
def change_color(u, X, Y, N, H, F, C, L):
    """Change the color of 'u' from X to Y and update N, H, F, C."""
    assert F[u] == X and X != Y
    F[u] = Y
    for k in C:
        if N[u, k] == 0:
            H[X, k] -= 1
            H[Y, k] += 1
    for v in L[u]:
        N[v, X] -= 1
        N[v, Y] += 1
        if N[v, X] == 0:
            H[F[v], X] += 1
        if N[v, Y] == 1:
            H[F[v], Y] -= 1
    C[X].remove(u)
    C[Y].append(u)