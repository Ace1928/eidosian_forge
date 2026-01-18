from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
@staticmethod
def interval_vector_mid_points(vec):
    """
        Given a vector of complex intervals, return the midpoints (as 0-length
        complex intervals) of them.
        """
    BaseField = vec[0].parent()
    return vec.apply_map(lambda shape: BaseField(shape.center()))