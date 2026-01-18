import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_stack_arrays_dispatcher)
def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False, autoconvert=False):
    """
    Superposes arrays fields by fields

    Parameters
    ----------
    arrays : array or sequence
        Sequence of input arrays.
    defaults : dictionary, optional
        Dictionary mapping field names to the corresponding default values.
    usemask : {True, False}, optional
        Whether to return a MaskedArray (or MaskedRecords is
        `asrecarray==True`) or a ndarray.
    asrecarray : {False, True}, optional
        Whether to return a recarray (or MaskedRecords if `usemask==True`)
        or just a flexible-type ndarray.
    autoconvert : {False, True}, optional
        Whether automatically cast the type of the field to the maximum.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> x = np.array([1, 2,])
    >>> rfn.stack_arrays(x) is x
    True
    >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
    >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
    ...   dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])
    >>> test = rfn.stack_arrays((z,zz))
    >>> test
    masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),
                       (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],
                 mask=[(False, False,  True), (False, False,  True),
                       (False, False, False), (False, False, False),
                       (False, False, False)],
           fill_value=(b'N/A', 1.e+20, 1.e+20),
                dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])

    """
    if isinstance(arrays, ndarray):
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    seqarrays = [np.asanyarray(a).ravel() for a in arrays]
    nrecords = [len(a) for a in seqarrays]
    ndtype = [a.dtype for a in seqarrays]
    fldnames = [d.names for d in ndtype]
    dtype_l = ndtype[0]
    newdescr = _get_fieldspec(dtype_l)
    names = [n for n, d in newdescr]
    for dtype_n in ndtype[1:]:
        for fname, fdtype in _get_fieldspec(dtype_n):
            if fname not in names:
                newdescr.append((fname, fdtype))
                names.append(fname)
            else:
                nameidx = names.index(fname)
                _, cdtype = newdescr[nameidx]
                if autoconvert:
                    newdescr[nameidx] = (fname, max(fdtype, cdtype))
                elif fdtype != cdtype:
                    raise TypeError("Incompatible type '%s' <> '%s'" % (cdtype, fdtype))
    if len(newdescr) == 1:
        output = ma.concatenate(seqarrays)
    else:
        output = ma.masked_all((np.sum(nrecords),), newdescr)
        offset = np.cumsum(np.r_[0, nrecords])
        seen = []
        for a, n, i, j in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
            names = a.dtype.names
            if names is None:
                output['f%i' % len(seen)][i:j] = a
            else:
                for name in n:
                    output[name][i:j] = a[name]
                    if name not in seen:
                        seen.append(name)
    return _fix_output(_fix_defaults(output, defaults), usemask=usemask, asrecarray=asrecarray)