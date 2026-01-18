import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def code_to_sign_sequence(code):
    """ Produces a sign sequence for a given Dowker code. """

    def first_non_zero(L):
        return min((i for i in range(len(L)) if L[i]))
    N = len(code)
    signs = [abs(n) - 1 for n in code]
    pairs = list(zip(range(0, 2 * N, 2), signs))
    pairs_dict = dict([(x, y) for x, y in pairs] + [(y, x) for x, y in pairs])
    full_code = [pairs_dict[i] for i in range(2 * N)]
    seq = full_code * 2
    emb, A = ([0] * 2 * N, [0] * 2 * N)
    A[0], A[seq[0]] = (1, 1)
    emb[0], emb[seq[0]] = (1, -1)
    all_phi = [[0] * 2 * N for i in range(2 * N)]
    for i in range(2 * N):
        all_phi[i][i] = 1
        for j in range(i, i + 2 * N):
            all_phi[i][j % (2 * N)] = 1 if i == j else -all_phi[i][(j - 1) % (2 * N)] if i <= seq[j] <= seq[i] else all_phi[i][(j - 1) % (2 * N)]
    while any(A):
        i = first_non_zero(A)
        D = [1] * 2 * N
        D[i:seq[i] + 1] = [0] * (seq[i] - i + 1)
        while any(D):
            x = first_non_zero(D)
            D[x] = 0
            if (i <= seq[x] <= seq[i] and emb[x] != 0 and (all_phi[i][x] * all_phi[i][seq[x]] * emb[i] != emb[x]) or ((seq[x] < i or seq[i] < seq[x]) and all_phi[i][x] * all_phi[i][seq[x]] != 1)) and x < i:
                raise ValueError('Not a realisable DT-code.')
            if seq[i] < seq[x] or seq[x] < i:
                D[seq[x]] = 0
            elif emb[x] == 0:
                assert D[seq[x]] == 0
                emb[x] = all_phi[i][x] * all_phi[i][seq[x]] * emb[i]
                emb[seq[x]] = -emb[x]
                if abs(seq[x] - seq[x - 1]) % (2 * N) != 1:
                    A[x] = 1
                    A[seq[x]] = 1
        A[i], A[seq[i]] = (0, 0)
    return [emb[i] for i in range(0, 2 * N, 2)]