from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
@staticmethod
def so3_matrix_and_derivative_for_z_rotation(angle):
    c = angle.cos()
    s = angle.sin()
    return (matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]]), matrix([[-s, -c, 0], [c, -s, 0], [0, 0, 0]]))