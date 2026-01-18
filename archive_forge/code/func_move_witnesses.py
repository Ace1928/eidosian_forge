from collections import defaultdict
import networkx as nx
def move_witnesses(src_color, dst_color, N, H, F, C, T_cal, L):
    """Move witness along a path from src_color to dst_color."""
    X = src_color
    while X != dst_color:
        Y = T_cal[X]
        w = next((x for x in C[X] if N[x, Y] == 0))
        change_color(w, X, Y, N=N, H=H, F=F, C=C, L=L)
        X = Y