from snappy.pari import pari
def is_pari_matrix(obj):
    return isinstance(obj, PariGen) and obj.type() == 't_MAT'