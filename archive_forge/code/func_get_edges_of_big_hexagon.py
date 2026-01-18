from snappy.snap import t3mlite as t3m
@staticmethod
def get_edges_of_big_hexagon(tet_and_perm):
    tet_index, p = tet_and_perm
    return [TruncatedComplex.Edge('beta', tet_and_perm), TruncatedComplex.Edge('alpha', (tet_index, p * t3m.Perm4([0, 2, 1, 3]))), TruncatedComplex.Edge('beta', (tet_index, p * t3m.Perm4([2, 0, 1, 3]))), TruncatedComplex.Edge('alpha', (tet_index, p * t3m.Perm4([2, 1, 0, 3]))), TruncatedComplex.Edge('beta', (tet_index, p * t3m.Perm4([1, 2, 0, 3]))), TruncatedComplex.Edge('alpha', (tet_index, p * t3m.Perm4([1, 0, 2, 3])))]