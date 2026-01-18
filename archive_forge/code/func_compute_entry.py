from ..pari import pari
import fractions
def compute_entry(i, j):
    return sum([m[i][k] * n[k][j] for k in range(num_cols_m)])