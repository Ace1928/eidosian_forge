from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def fill_dcpl(plist, shape, dtype, chunks, compression, compression_opts, shuffle, fletcher32, maxshape, scaleoffset, external, allow_unknown_filter=False):
    """ Generate a dataset creation property list.

    Undocumented and subject to change without warning.
    """
    if shape is None or shape == ():
        shapetype = 'Empty' if shape is None else 'Scalar'
        if any((chunks, compression, compression_opts, shuffle, fletcher32, scaleoffset is not None)):
            raise TypeError(f"{shapetype} datasets don't support chunk/filter options")
        if maxshape and maxshape != ():
            raise TypeError(f'{shapetype} datasets cannot be extended')
        return h5p.create(h5p.DATASET_CREATE)

    def rq_tuple(tpl, name):
        """ Check if chunks/maxshape match dataset rank """
        if tpl in (None, True):
            return
        try:
            tpl = tuple(tpl)
        except TypeError:
            raise TypeError('"%s" argument must be None or a sequence object' % name)
        if len(tpl) != len(shape):
            raise ValueError('"%s" must have same rank as dataset shape' % name)
    rq_tuple(chunks, 'chunks')
    rq_tuple(maxshape, 'maxshape')
    if compression is not None:
        if isinstance(compression, FilterRefBase):
            compression_opts = compression.filter_options
            compression = compression.filter_id
        if compression not in encode and (not isinstance(compression, int)):
            raise ValueError('Compression filter "%s" is unavailable' % compression)
        if compression == 'gzip':
            if compression_opts is None:
                gzip_level = DEFAULT_GZIP
            elif compression_opts in range(10):
                gzip_level = compression_opts
            else:
                raise ValueError('GZIP setting must be an integer from 0-9, not %r' % compression_opts)
        elif compression == 'lzf':
            if compression_opts is not None:
                raise ValueError('LZF compression filter accepts no options')
        elif compression == 'szip':
            if compression_opts is None:
                compression_opts = DEFAULT_SZIP
            err = "SZIP options must be a 2-tuple ('ec'|'nn', even integer 0-32)"
            try:
                szmethod, szpix = compression_opts
            except TypeError:
                raise TypeError(err)
            if szmethod not in ('ec', 'nn'):
                raise ValueError(err)
            if not (0 < szpix <= 32 and szpix % 2 == 0):
                raise ValueError(err)
    elif compression_opts is not None:
        raise TypeError('Compression method must be specified')
    if scaleoffset is not None:
        if scaleoffset < 0:
            raise ValueError('scale factor must be >= 0')
        if dtype.kind == 'f':
            if scaleoffset is True:
                raise ValueError('integer scaleoffset must be provided for floating point types')
        elif dtype.kind in ('u', 'i'):
            if scaleoffset is True:
                scaleoffset = h5z.SO_INT_MINBITS_DEFAULT
        else:
            raise TypeError('scale/offset filter only supported for integer and floating-point types')
        if fletcher32:
            raise ValueError('fletcher32 cannot be used with potentially lossy scale/offset filter')
    external = _normalize_external(external)
    if chunks is True or (chunks is None and any((shuffle, fletcher32, compression, maxshape, scaleoffset is not None))):
        chunks = guess_chunk(shape, maxshape, dtype.itemsize)
    if maxshape is True:
        maxshape = (None,) * len(shape)
    if chunks is not None:
        plist.set_chunk(chunks)
        plist.set_fill_time(h5d.FILL_TIME_ALLOC)
    if scaleoffset is not None:
        if dtype.kind in ('u', 'i'):
            plist.set_scaleoffset(h5z.SO_INT, scaleoffset)
        else:
            plist.set_scaleoffset(h5z.SO_FLOAT_DSCALE, scaleoffset)
    for item in external:
        plist.set_external(*item)
    if shuffle:
        plist.set_shuffle()
    if compression == 'gzip':
        plist.set_deflate(gzip_level)
    elif compression == 'lzf':
        plist.set_filter(h5z.FILTER_LZF, h5z.FLAG_OPTIONAL)
    elif compression == 'szip':
        opts = {'ec': h5z.SZIP_EC_OPTION_MASK, 'nn': h5z.SZIP_NN_OPTION_MASK}
        plist.set_szip(opts[szmethod], szpix)
    elif isinstance(compression, int):
        if not allow_unknown_filter and (not h5z.filter_avail(compression)):
            raise ValueError('Unknown compression filter number: %s' % compression)
        plist.set_filter(compression, h5z.FLAG_OPTIONAL, compression_opts)
    if fletcher32:
        plist.set_fletcher32()
    return plist