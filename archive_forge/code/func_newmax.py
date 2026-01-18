import itertools
from future import utils
def newmax(*args, **kwargs):
    return new_min_max(_builtin_max, *args, **kwargs)