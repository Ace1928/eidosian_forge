from snappy.snap import t3mlite as t3m
@staticmethod
def get_first_tet_and_odd_perm_for_vertex(vertex):
    for tet_and_perm in TruncatedComplex.get_tet_and_odd_perms_for_vertex(vertex):
        return tet_and_perm