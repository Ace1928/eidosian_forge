import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def read_subheaders(fileobj, mlist, endianness):
    """Retrieve all subheaders and return list of subheader recarrays

    Parameters
    ----------
    fileobj : file-like
        implementing ``read`` and ``seek``
    mlist : (nframes, 4) ndarray
        Columns are:
        * 0 - Matrix identifier.
        * 1 - subheader block number
        * 2 - Last block number of matrix data block.
        * 3 - Matrix status
    endianness : {'<', '>'}
        little / big endian code

    Returns
    -------
    subheaders : list
        List of subheader structured arrays
    """
    subheaders = []
    dt = subhdr_dtype
    if endianness is not native_code:
        dt = dt.newbyteorder(endianness)
    for mat_id, sh_blkno, sh_last_blkno, mat_stat in mlist:
        if sh_blkno == 0:
            break
        offset = (sh_blkno - 1) * BLOCK_SIZE
        fileobj.seek(offset)
        tmpdat = fileobj.read(BLOCK_SIZE)
        sh = np.ndarray(shape=(), dtype=dt, buffer=tmpdat)
        subheaders.append(sh)
    return subheaders