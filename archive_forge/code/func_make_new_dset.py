import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
def make_new_dset(parent, shape=None, dtype=None, data=None, name=None, chunks=None, compression=None, shuffle=None, fletcher32=None, maxshape=None, compression_opts=None, fillvalue=None, scaleoffset=None, track_times=False, external=None, track_order=None, dcpl=None, dapl=None, efile_prefix=None, virtual_prefix=None, allow_unknown_filter=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None):
    """ Return a new low-level dataset identifier """
    if data is not None and (not isinstance(data, Empty)):
        data = array_for_new_object(data, specified_dtype=dtype)
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError('One of data, shape or dtype must be specified')
            data = Empty(dtype)
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and product(shape) != product(data.shape):
            raise ValueError('Shape tuple is incompatible with data')
    if isinstance(maxshape, int):
        maxshape = (maxshape,)
    tmp_shape = maxshape if maxshape is not None else shape
    if isinstance(chunks, int) and (not isinstance(chunks, bool)):
        chunks = (chunks,)
    if isinstance(chunks, tuple) and any((chunk > dim for dim, chunk in zip(tmp_shape, chunks) if dim is not None)):
        errmsg = 'Chunk shape must not be greater than data shape in any dimension. {} is not compatible with {}'.format(chunks, shape)
        raise ValueError(errmsg)
    if isinstance(dtype, Datatype):
        tid = dtype.id
        dtype = tid.dtype
    else:
        if dtype is None and data is None:
            dtype = numpy.dtype('=f4')
        elif dtype is None and data is not None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)
        tid = h5t.py_create(dtype, logical=1)
    if any((compression, shuffle, fletcher32, maxshape, scaleoffset)) and chunks is False:
        raise ValueError('Chunked format required for given storage options')
    if compression is True:
        if compression_opts is None:
            compression_opts = 4
        compression = 'gzip'
    if compression in _LEGACY_GZIP_COMPRESSION_VALS:
        if compression_opts is not None:
            raise TypeError('Conflict in compression options')
        compression_opts = compression
        compression = 'gzip'
    dcpl = filters.fill_dcpl(dcpl or h5p.create(h5p.DATASET_CREATE), shape, dtype, chunks, compression, compression_opts, shuffle, fletcher32, maxshape, scaleoffset, external, allow_unknown_filter)
    if fillvalue is not None:
        string_info = h5t.check_string_dtype(dtype)
        if string_info is not None:
            dtype = h5t.string_dtype(string_info.encoding)
            fillvalue = numpy.array(fillvalue, dtype=dtype)
        else:
            fillvalue = numpy.array(fillvalue)
        dcpl.set_fill_value(fillvalue)
    if track_times is None:
        track_times = False
    if track_times in (True, False):
        dcpl.set_obj_track_times(track_times)
    else:
        raise TypeError('track_times must be either True or False')
    if track_order is True:
        dcpl.set_attr_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    elif track_order is False:
        dcpl.set_attr_creation_order(0)
    elif track_order is not None:
        raise TypeError('track_order must be either True or False')
    if maxshape is not None:
        maxshape = tuple((m if m is not None else h5s.UNLIMITED for m in maxshape))
    if any([efile_prefix, virtual_prefix, rdcc_nbytes, rdcc_nslots, rdcc_w0]):
        dapl = dapl or h5p.create(h5p.DATASET_ACCESS)
    if efile_prefix is not None:
        dapl.set_efile_prefix(efile_prefix)
    if virtual_prefix is not None:
        dapl.set_virtual_prefix(virtual_prefix)
    if rdcc_nbytes or rdcc_nslots or rdcc_w0:
        cache_settings = list(dapl.get_chunk_cache())
        if rdcc_nslots is not None:
            cache_settings[0] = rdcc_nslots
        if rdcc_nbytes is not None:
            cache_settings[1] = rdcc_nbytes
        if rdcc_w0 is not None:
            cache_settings[2] = rdcc_w0
        dapl.set_chunk_cache(*cache_settings)
    if isinstance(data, Empty):
        sid = h5s.create(h5s.NULL)
    else:
        sid = h5s.create_simple(shape, maxshape)
    dset_id = h5d.create(parent.id, name, tid, sid, dcpl=dcpl, dapl=dapl)
    if data is not None and (not isinstance(data, Empty)):
        dset_id.write(h5s.ALL, h5s.ALL, data)
    return dset_id