import math
def poincare_stl(face_dicts, num_subdivisions=5, cutoff_radius=0.9):
    """ Yield the output of klein_stl(face_dicts, ...) after applying projection to every vertex produced. """
    for triangle in subdivide_triangles(klein_stl(face_dicts), num_subdivisions):
        yield (projection(triangle[0], cutoff_radius), projection(triangle[1], cutoff_radius), projection(triangle[2], cutoff_radius))
    return