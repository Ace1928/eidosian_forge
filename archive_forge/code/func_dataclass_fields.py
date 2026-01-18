from collections import OrderedDict
from collections.abc import Iterator
from operator import getitem
import uuid
import ray
from dask.base import quote
from dask.core import get as get_sync
from dask.utils import apply
def dataclass_fields(x):
    return []