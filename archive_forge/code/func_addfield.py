from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def addfield(mrecord, newfield, newfieldname=None):
    """Adds a new field to the masked record array

    Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
    is None, the new field name is set to 'fi', where `i` is the number of
    existing fields.

    """
    _data = mrecord._data
    _mask = mrecord._mask
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = 'f%i' % len(_data.dtype)
    newfield = ma.array(newfield)
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    newdata = recarray(_data.shape, newdtype)
    [newdata.setfield(_data.getfield(*f), *f) for f in _data.dtype.fields.values()]
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    newmdtype = np.dtype([(n, bool_) for n in newdtype.names])
    newmask = recarray(_data.shape, newmdtype)
    [newmask.setfield(_mask.getfield(*f), *f) for f in _mask.dtype.fields.values()]
    newmask.setfield(getmaskarray(newfield), *newmask.dtype.fields[newfieldname])
    newdata._mask = newmask
    return newdata