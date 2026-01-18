from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def rational_cohomology_basis(trig):
    """
    Given a SnapPy triangulation, give 2-cocycles encoded as weights
    per face per tetrahedron that generate the second rational cohomology
    group.

        >>> from snappy import Manifold
        >>> rational_cohomology_basis(Manifold("m125"))
        [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]]

    """
    signs_and_face_class_indices = compute_signs_and_face_class_indices(trig)
    matrix2, edge_labels, face_labels = trig._ptolemy_equations_boundary_map_2()
    matrix3, face_labels, tet_labels = trig._ptolemy_equations_boundary_map_3()
    two_cocycles_and_orders = homology_basis_representatives_with_orders(matrix2, matrix3, 0)
    rational_two_cocycles = [two_cocycle for two_cocycle, order in two_cocycles_and_orders if order == 0]
    return [[sgn * two_cocycle[index] for sgn, index in signs_and_face_class_indices] for two_cocycle in rational_two_cocycles]