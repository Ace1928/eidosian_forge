from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def check_weights_valid(trig, weights):
    """
    Given a SnapPy triangulation and weights per face per tetrahedron,
    check they are consistent across glued faces and form a 2-cocycle.

        >>> from snappy import Manifold
        >>> check_weights_valid(Manifold("m015"), [1, 0, 0, 0, -1, 0, 0, 1, -1, 0, 0, 0])
        >>> check_weights_valid(Manifold("m004"), [0, 1, 0, 1, 0, 0, -1, -2])
        Traceback (most recent call last):
           ...
        ValueError: Weights for identified faces do not match
        >>> check_weights_valid(Manifold("m003"), [1, 0, 0, 2, -1, 0, 0, -2])
        Traceback (most recent call last):
           ...
        ValueError: Weights are not a 2-cocycle.

    """
    face_classes = trig._ptolemy_equations_identified_face_classes()
    check_face_class_weights_valid(trig, [value_for_face_class(weights, face_class) for face_class in face_classes])