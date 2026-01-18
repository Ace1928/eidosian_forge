import itertools
from future import utils
def newmin(*args, **kwargs):
    return new_min_max(_builtin_min, *args, **kwargs)