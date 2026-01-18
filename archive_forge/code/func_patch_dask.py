import dask
from .scheduler import (
from .callbacks import (
from .optimizations import dataframe_optimize
def patch_dask(ray_dask_persist, ray_dask_persist_mixin):
    dask.persist = ray_dask_persist
    dask.base.DaskMethodsMixin.persist = ray_dask_persist_mixin