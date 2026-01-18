from collections import OrderedDict
from collections.abc import Iterator
from operator import getitem
import uuid
import ray
from dask.base import quote
from dask.core import get as get_sync
from dask.utils import apply
def repack(results):
    dsk = repack_dsk.copy()
    dsk[object_refs_token] = quote(results)
    return get_sync(dsk, out)