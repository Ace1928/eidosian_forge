from contextlib import contextmanager
import posixpath as pp
import numpy
from .compat import filename_decode, filename_encode
from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support
def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
    """ Create a new HDF5 dataset

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.

        Keyword-only arguments:

        chunks
            (Tuple or int) Chunk shape, or True to enable auto-chunking. Integers can
            be used for 1D shape.

        maxshape
            (Tuple or int) Make the dataset resizable up to this shape. Use None for
            axes you want to be unlimited. Integers can be used for 1D shape.
        compression
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
        scaleoffset
            (Integer) Enable scale/offset filter for (usually) lossy
            compression of integer or floating-point data. For integer
            data, the value of scaleoffset is the number of bits to
            retain (pass 0 to let HDF5 determine the minimum number of
            bits necessary for lossless compression). For floating point
            data, scaleoffset is the number of digits after the decimal
            place to retain; stored values thus have absolute error
            less than 0.5*10**(-scaleoffset).
        shuffle
            (T/F) Enable shuffle filter.
        fletcher32
            (T/F) Enable fletcher32 error detection. Not permitted in
            conjunction with the scale/offset filter.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        track_times
            (T/F) Enable dataset creation timestamps.
        track_order
            (T/F) Track attribute creation order if True. If omitted use
            global default h5.get_config().track_order.
        external
            (Iterable of tuples) Sets the external storage property, thus
            designating that the dataset will be stored in one or more
            non-HDF5 files external to the HDF5 file.  Adds each tuple
            of (name, offset, size) to the dataset's list of external files.
            Each name must be a str, bytes, or os.PathLike; each offset and
            size, an integer.  If only a name is given instead of an iterable
            of tuples, it is equivalent to [(name, 0, h5py.h5f.UNLIMITED)].
        efile_prefix
            (String) External dataset file prefix for dataset access property
            list. Does not persist in the file.
        virtual_prefix
            (String) Virtual dataset file prefix for dataset access property
            list. Does not persist in the file.
        allow_unknown_filter
            (T/F) Do not check that the requested filter is available for use.
            This should only be used with ``write_direct_chunk``, where the caller
            compresses the data before handing it to h5py.
        rdcc_nbytes
            Total size of the dataset's chunk cache in bytes. The default size
            is 1024**2 (1 MiB).
        rdcc_w0
            The chunk preemption policy for this dataset.  This must be
            between 0 and 1 inclusive and indicates the weighting according to
            which chunks which have been fully read or written are penalized
            when determining which chunks to flush from cache.  A value of 0
            means fully read or written chunks are treated no differently than
            other chunks (the preemption is strictly LRU) while a value of 1
            means fully read or written chunks are always preempted before
            other chunks.  If your application only reads or writes data once,
            this can be safely set to 1.  Otherwise, this should be set lower
            depending on how often you re-read or re-write the same data.  The
            default value is 0.75.
        rdcc_nslots
            The number of chunk slots in the dataset's chunk cache. Increasing
            this value reduces the number of cache collisions, but slightly
            increases the memory used. Due to the hashing strategy, this value
            should ideally be a prime number. As a rule of thumb, this value
            should be at least 10 times the number of chunks that can fit in
            rdcc_nbytes bytes. For maximum performance, this value should be set
            approximately 100 times that number of chunks. The default value is
            521.
        """
    if 'track_order' not in kwds:
        kwds['track_order'] = h5.get_config().track_order
    if 'efile_prefix' in kwds:
        kwds['efile_prefix'] = self._e(kwds['efile_prefix'])
    if 'virtual_prefix' in kwds:
        kwds['virtual_prefix'] = self._e(kwds['virtual_prefix'])
    with phil:
        group = self
        if name:
            name = self._e(name)
            if b'/' in name.lstrip(b'/'):
                parent_path, name = name.rsplit(b'/', 1)
                group = self.require_group(parent_path)
        dsid = dataset.make_new_dset(group, shape, dtype, data, name, **kwds)
        dset = dataset.Dataset(dsid)
        return dset