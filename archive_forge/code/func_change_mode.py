from itertools import chain
from typing import Any, Callable, Iterable, Optional
import numpy
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
import cupy
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._core._routines_math as _math
import cupy._core._routines_statistics as _statistics
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _linalg
def change_mode(self, mode: _modes.Mode) -> 'DistributedArray':
    """Return a view or a copy in the given mode.

        Args:
            mode (mode Object): How overlaps of
                the chunks are interpreted.

        .. seealso::
                :attr:`DistributedArray.mode` for details about modes.
        """
    return self._to_op_mode(mode)