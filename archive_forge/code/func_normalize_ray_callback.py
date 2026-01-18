import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def normalize_ray_callback(cb):
    if isinstance(cb, RayDaskCallback):
        return cb._ray_callback
    elif isinstance(cb, RayCallback):
        return cb
    else:
        raise TypeError("Callbacks must be either 'RayDaskCallback' or 'RayCallback' namedtuple")