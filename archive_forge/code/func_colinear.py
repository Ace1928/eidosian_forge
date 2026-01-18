from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def colinear(u, v, w):
    return are_parallel(v - u, w - u)