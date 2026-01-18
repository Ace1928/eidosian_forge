from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def norm_sq(v):
    return sum((x ** 2 for x in v))