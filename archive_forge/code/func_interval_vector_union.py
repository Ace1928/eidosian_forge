from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RealDoubleField, RealIntervalField, vector, matrix, pi
@staticmethod
def interval_vector_union(vecA, vecB):
    """
        Given two vectors of intervals, return the vector of their unions,
        i.e., the smallest interval containing both intervals.
        """
    return vector([a.union(b) for a, b in zip(vecA, vecB)])