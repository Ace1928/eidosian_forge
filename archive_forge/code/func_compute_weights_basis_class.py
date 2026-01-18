from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def compute_weights_basis_class(trig, cohomology_class):
    """
    Convenience method to quickly access cohomology classes for
    M.inside_view().

        >>> from snappy import Manifold
        >>> compute_weights_basis_class(Manifold("m004"), None)
        (None, None, None)
        >>> compute_weights_basis_class(Manifold("m003"), [1, 0, 0, 1, -1, 0, 0, -1])
        ([1, 0, 0, 1, -1, 0, 0, -1], None, None)
        >>> compute_weights_basis_class(Manifold("m003"), 0)
        (None, [[1, 0, 0, 1, -1, 0, 0, -1]], [1.0])
        >>> compute_weights_basis_class(Manifold("m125"), 0)
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [1.0, 0.0])
        >>> compute_weights_basis_class(Manifold("m125"), 1)
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [0.0, 1.0])
        >>> compute_weights_basis_class(Manifold("m125"), [0.5, 0.5])
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [0.5, 0.5])

    """
    if cohomology_class is None:
        return (None, None, None)
    try:
        as_list = list(cohomology_class)
    except TypeError:
        as_list = None
    if as_list:
        if len(as_list) == 4 * trig.num_tetrahedra():
            check_weights_valid(trig, as_list)
            return (as_list, None, None)
        basis = rational_cohomology_basis(trig)
        if len(as_list) == len(basis):
            return (None, basis, as_list)
        raise ValueError('Expected array of length %d or %d either assigning one number for each basis vector of the second rational cohomology group or one weight per face and tetrahedron.' % (len(basis), 4 * trig.num_tetrahedra()))
    else:
        basis = rational_cohomology_basis(trig)
        c = range(len(basis))[cohomology_class]
        return (None, basis, [1.0 if i == c else 0.0 for i in range(len(basis))])