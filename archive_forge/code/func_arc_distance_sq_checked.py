from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def arc_distance_sq_checked(arc_a0, arc_b0):
    """
    Check that all possible permutations of the data specifying the
    arcs gives the same result, plus images under some orthogonal
    transformations.
    """
    ans = arc_distance_sq(arc_a0, arc_b0)
    arc_a1 = (arc_a0[1], arc_a0[0])
    arc_b1 = (arc_b0[1], arc_b0[0])
    mats = [pall_matrix(1, 0, 0, 0), pall_matrix(1, -2, 3, 5), pall_matrix(4, -1, 2, 2)]
    for M in mats:
        arc_a0 = (M * arc_a0[0], M * arc_a0[1])
        arc_a1 = (M * arc_a1[0], M * arc_a1[1])
        arc_b0 = (M * arc_b0[0], M * arc_b0[1])
        arc_b1 = (M * arc_b1[0], M * arc_b1[1])
        assert ans == arc_distance_sq(arc_a0, arc_b0)
        assert ans == arc_distance_sq(arc_a0, arc_b1)
        assert ans == arc_distance_sq(arc_a1, arc_b0)
        assert ans == arc_distance_sq(arc_a1, arc_b1)
        assert ans == arc_distance_sq(arc_b0, arc_a0)
        assert ans == arc_distance_sq(arc_b0, arc_a1)
        assert ans == arc_distance_sq(arc_b1, arc_a0)
        assert ans == arc_distance_sq(arc_b1, arc_a1)
    return ans