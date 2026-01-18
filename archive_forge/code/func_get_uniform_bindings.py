from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def get_uniform_bindings(self):
    d = super(FiniteRaytracingData, self).get_uniform_bindings()
    d['TetrahedraEdges.R13EdgeEnds'] = ('vec4[]', [edge_end for tet in self.mcomplex.Tetrahedra for E in t3m.OneSubsimplices for edge_end in tet.R13_edge_ends[E]])
    d['isNonGeometric'] = ('bool', False)
    d['nonGeometricTexture'] = ('int', 0)
    return d