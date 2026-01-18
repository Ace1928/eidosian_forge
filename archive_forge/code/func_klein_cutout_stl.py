import math
def klein_cutout_stl(face_dicts, shrink_factor=0.9):
    """ Yield triangles describing these faces after removing a fraction of the interior.

    The fraction removed is given by shrink_factor. """
    for face in face_dicts:
        vertices = face['vertices']
        center = [sum((vertex[i] for vertex in vertices)) / len(vertices) for i in range(3)]
        new_vertices = [[vertex[i] + (center[i] - vertex[i]) / 3 for i in range(3)] for vertex in vertices]
        new_inside_points = [[point[i] * shrink_factor for i in range(3)] for point in new_vertices]
        for i in range(len(new_vertices)):
            yield (new_vertices[i], new_inside_points[(i + 1) % len(new_vertices)], new_inside_points[i])
            yield (new_vertices[i], new_vertices[(i + 1) % len(new_vertices)], new_inside_points[(i + 1) % len(new_vertices)])
            yield (vertices[i], new_vertices[(i + 1) % len(vertices)], new_vertices[i])
            yield (vertices[i], vertices[(i + 1) % len(vertices)], new_vertices[(i + 1) % len(vertices)])
            yield tuple((tuple((shrink_factor * coord for coord in point)) for point in (vertices[i], new_vertices[i], new_vertices[(i + 1) % len(vertices)])))
            yield tuple((tuple((shrink_factor * coord for coord in point)) for point in (vertices[i], new_vertices[(i + 1) % len(vertices)], vertices[(i + 1) % len(vertices)])))
    return