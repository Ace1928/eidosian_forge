import numpy as np
from matplotlib import _api
@_api.deprecated('3.8')
def persp_transformation(zfront, zback, focal_length):
    return _persp_transformation(zfront, zback, focal_length)