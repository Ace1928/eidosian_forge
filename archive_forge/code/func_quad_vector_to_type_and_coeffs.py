import snappy
import FXrays
def quad_vector_to_type_and_coeffs(quad_vector):
    """
    For an n-tetrahedra manifold, take a full quad vector
    of length 3n and store the quad type and weight for
    each tetrahedron.
    """
    quad_types, coefficients = ([], [])
    quad_vector = list(quad_vector)
    for i in range(len(quad_vector) // 3):
        one_tet = quad_vector[3 * i:3 * (i + 1)]
        pos = [(i, c) for i, c in enumerate(one_tet) if c > 0]
        assert len(pos) <= 1
        if pos:
            q, c = pos[0]
            quad_types.append(q)
            coefficients.append(c)
        else:
            quad_types.append(None)
            coefficients.append(0)
    return (quad_types, Vector(coefficients))