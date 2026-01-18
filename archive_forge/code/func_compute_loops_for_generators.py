from . import matrix
def compute_loops_for_generators(M, penalties):
    """
    Given a SnapPy Manifold M, return a loop of short, middle, and long edges
    representing a generator of the fundamental group in the
    unsimplified presentation for each generator.

    Each short, middle, respectively, long edge has an associate penalty
    (encoded the triple penalties). For each generator, the method returns
    a loop with the smallest total penalty.
    """
    M._choose_generators(False, False)
    choose_generators_info = M._choose_generators_info()
    point_identification_dict = _compute_point_identification_dict(choose_generators_info)
    origin = _compute_origin(choose_generators_info)
    point_to_shortest_path = _compute_point_to_shortest_path(point_identification_dict, origin, penalties)
    return _compute_loops_for_generators_from_info(choose_generators_info, point_to_shortest_path, penalties)