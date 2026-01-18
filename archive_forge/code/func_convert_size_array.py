import math
import warnings
import matplotlib.dates
def convert_size_array(size_array):
    size = [math.sqrt(s) for s in size_array]
    if len(size) == 1:
        return size[0]
    else:
        return size