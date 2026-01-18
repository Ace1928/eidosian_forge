import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError
import numpy as np
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
def read_data_block(darray, fname, data, mmap):
    """Parses data from a <Data> element, or loads from an external file.

    Parameters
    ----------
    darray : GiftiDataArray
         GiftiDataArray object representing the parent <DataArray> of this
         <Data> element

    fname : str or None
         Name of GIFTI file being loaded, or None if in-memory

    data : str or None
         Data to parse, or None if data is in an external file

    mmap : {True, False, 'c', 'r', 'r+'}
        Controls the use of numpy memory mapping for reading data.  Only has
        an effect when loading GIFTI images with data stored in external files
        (``DataArray`` elements with an ``Encoding`` equal to
        ``ExternalFileBinary``).  If ``False``, do not try numpy ``memmap``
        for data array.  If one of ``{'c', 'r', 'r+'}``, try numpy ``memmap``
        with ``mode=mmap``.  A `mmap` value of ``True`` gives the same
        behavior as ``mmap='c'``.  If the file cannot be memory-mapped, ignore
        `mmap` value and read array from file.

    Returns
    -------
    ``numpy.ndarray`` or ``numpy.memmap`` containing the parsed data
    """
    if mmap not in (True, False, 'c', 'r', 'r+'):
        raise ValueError("mmap value should be one of True, False, 'c', 'r', 'r+'")
    if mmap is True:
        mmap = 'c'
    enclabel = gifti_encoding_codes.label[darray.encoding]
    if enclabel not in ('ASCII', 'B64BIN', 'B64GZ', 'External'):
        raise GiftiParseError(f'Unknown encoding {darray.encoding}')
    byteorder = gifti_endian_codes.byteorder[darray.endian]
    dtype = data_type_codes.dtype[darray.datatype].newbyteorder(byteorder)
    shape = tuple(darray.dims)
    order = array_index_order_codes.npcode[darray.ind_ord]
    if enclabel == 'ASCII':
        return np.loadtxt(StringIO(data), dtype=dtype, ndmin=1).reshape(shape, order=order)
    if enclabel == 'External':
        if fname is None:
            raise GiftiParseError('ExternalFileBinary is not supported when loading from in-memory XML')
        ext_fname = op.join(op.dirname(fname), darray.ext_fname)
        if not op.exists(ext_fname):
            raise GiftiParseError('Cannot locate external file ' + ext_fname)
        newarr = None
        if mmap:
            try:
                return np.memmap(ext_fname, dtype=dtype, mode=mmap, offset=darray.ext_offset, shape=shape, order=order)
            except (AttributeError, TypeError, ValueError):
                pass
        if newarr is None:
            return np.fromfile(ext_fname, dtype=dtype, count=np.prod(darray.dims), offset=darray.ext_offset).reshape(shape, order=order)
    dec = base64.b64decode(data.encode('ascii'))
    if enclabel == 'B64BIN':
        buff = bytearray(dec)
    else:
        buff = bytearray(zlib.decompress(dec))
    del dec
    return np.frombuffer(buff, dtype=dtype).reshape(shape, order=order)