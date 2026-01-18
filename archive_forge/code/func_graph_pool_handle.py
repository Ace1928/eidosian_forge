import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
from torch._C import (  # noqa: F401
def graph_pool_handle():
    """Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return _graph_pool_handle()