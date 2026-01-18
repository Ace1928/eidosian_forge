from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def get_compile_time_constants(self):
    d = super(FiniteRaytracingData, self).get_compile_time_constants()
    d[b'##finiteTrig##'] = 1
    return d