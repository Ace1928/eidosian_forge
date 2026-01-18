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
def require_dataset(self, name, shape, dtype, exact=False, **kwds):
    """ Open a dataset, creating it if it doesn't exist.

        If keyword "exact" is False (default), an existing dataset must have
        the same shape and a conversion-compatible dtype to be returned.  If
        True, the shape and dtype must match exactly.

        If keyword "maxshape" is given, the maxshape and dtype must match
        instead.

        If any of the keywords "rdcc_nslots", "rdcc_nbytes", or "rdcc_w0" are
        given, they will be used to configure the dataset's chunk cache.

        Other dataset keywords (see create_dataset) may be provided, but are
        only used if a new dataset is to be created.

        Raises TypeError if an incompatible object already exists, or if the
        shape, maxshape or dtype don't match according to the above rules.
        """
    if 'efile_prefix' in kwds:
        kwds['efile_prefix'] = self._e(kwds['efile_prefix'])
    if 'virtual_prefix' in kwds:
        kwds['virtual_prefix'] = self._e(kwds['virtual_prefix'])
    with phil:
        if name not in self:
            return self.create_dataset(name, *(shape, dtype), **kwds)
        if isinstance(shape, int):
            shape = (shape,)
        try:
            dsid = dataset.open_dset(self, self._e(name), **kwds)
            dset = dataset.Dataset(dsid)
        except KeyError:
            dset = self[name]
            raise TypeError('Incompatible object (%s) already exists' % dset.__class__.__name__)
        if shape != dset.shape:
            if 'maxshape' not in kwds:
                raise TypeError('Shapes do not match (existing %s vs new %s)' % (dset.shape, shape))
            elif kwds['maxshape'] != dset.maxshape:
                raise TypeError('Max shapes do not match (existing %s vs new %s)' % (dset.maxshape, kwds['maxshape']))
        if exact:
            if dtype != dset.dtype:
                raise TypeError('Datatypes do not exactly match (existing %s vs new %s)' % (dset.dtype, dtype))
        elif not numpy.can_cast(dtype, dset.dtype):
            raise TypeError('Datatypes cannot be safely cast (existing %s vs new %s)' % (dset.dtype, dtype))
        return dset