import math
def poincare_cutout_stl(face_dicts, num_subdivisions=3, shrink_factor=0.9, cutoff_radius=0.9):
    """ Yield the output of klein_cutout_stl(face_dicts, ...) after applying projection to every vertex produced. """
    for triangle in subdivide_triangles(klein_cutout_stl(face_dicts, shrink_factor), num_subdivisions):
        yield (projection(triangle[0], cutoff_radius), projection(triangle[1], cutoff_radius), projection(triangle[2], cutoff_radius))
    return