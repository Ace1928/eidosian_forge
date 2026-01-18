import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def regina_boundary_slope(surface):
    slope = surface.boundaryIntersections()
    a = int(slope.entry(0, 0).stringValue())
    b = int(slope.entry(0, 1).stringValue())
    return (b, -a)