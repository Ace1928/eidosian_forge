import math
import warnings
import matplotlib.dates
def get_axis_mirror(main_spine, mirror_spine):
    if main_spine and mirror_spine:
        return 'ticks'
    elif main_spine and (not mirror_spine):
        return False
    elif not main_spine and mirror_spine:
        return False
    else:
        return False