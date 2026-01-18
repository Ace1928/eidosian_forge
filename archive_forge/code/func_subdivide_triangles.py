import math
def subdivide_triangles(triangles, num_subdivisions):
    if num_subdivisions == 0:
        for triangle in triangles:
            yield triangle
    elif num_subdivisions == 1:
        midpoint = lambda P, Q: ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2, (P[2] + Q[2]) / 2)
        for x, y, z in triangles:
            yield (x, midpoint(x, y), midpoint(x, z))
            yield (midpoint(y, x), y, midpoint(y, z))
            yield (midpoint(z, x), midpoint(z, y), z)
            yield (midpoint(x, y), midpoint(y, z), midpoint(z, x))
    else:
        for triangle in subdivide_triangles(subdivide_triangles(triangles, 1), num_subdivisions - 1):
            yield triangle
    return