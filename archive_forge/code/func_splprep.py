import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None, full_output=0, nest=None, per=0, quiet=1):
    if task <= 0:
        _parcur_cache = {'t': array([], float), 'wrk': array([], float), 'iwrk': array([], dfitpack_int), 'u': array([], float), 'ub': 0, 'ue': 1}
    x = atleast_1d(x)
    idim, m = x.shape
    if per:
        for i in range(idim):
            if x[i][0] != x[i][-1]:
                if not quiet:
                    warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' % (i, m, i)), stacklevel=2)
                x[i][-1] = x[i][0]
    if not 0 < idim < 11:
        raise TypeError('0 < idim < 11 must hold')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    ipar = u is not None
    if ipar:
        _parcur_cache['u'] = u
        if ub is None:
            _parcur_cache['ub'] = u[0]
        else:
            _parcur_cache['ub'] = ub
        if ue is None:
            _parcur_cache['ue'] = u[-1]
        else:
            _parcur_cache['ue'] = ue
    else:
        _parcur_cache['u'] = zeros(m, float)
    if not 1 <= k <= 5:
        raise TypeError('1 <= k= %d <=5 must hold' % k)
    if not -1 <= task <= 1:
        raise TypeError('task must be -1, 0 or 1')
    if not len(w) == m or (ipar == 1 and (not len(u) == m)):
        raise TypeError('Mismatch of input dimensions')
    if s is None:
        s = m - sqrt(2 * m)
    if t is None and task == -1:
        raise TypeError('Knots must be given for task=-1')
    if t is not None:
        _parcur_cache['t'] = atleast_1d(t)
    n = len(_parcur_cache['t'])
    if task == -1 and n < 2 * k + 2:
        raise TypeError('There must be at least 2*k+2 knots for task=-1')
    if m <= k:
        raise TypeError('m > k must hold')
    if nest is None:
        nest = m + 2 * k
    if task >= 0 and s == 0 or nest < 0:
        if per:
            nest = m + 2 * k
        else:
            nest = m + k + 1
    nest = max(nest, 2 * k + 3)
    u = _parcur_cache['u']
    ub = _parcur_cache['ub']
    ue = _parcur_cache['ue']
    t = _parcur_cache['t']
    wrk = _parcur_cache['wrk']
    iwrk = _parcur_cache['iwrk']
    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k, task, ipar, s, t, nest, wrk, iwrk, per)
    _parcur_cache['u'] = o['u']
    _parcur_cache['ub'] = o['ub']
    _parcur_cache['ue'] = o['ue']
    _parcur_cache['t'] = t
    _parcur_cache['wrk'] = o['wrk']
    _parcur_cache['iwrk'] = o['iwrk']
    ier = o['ier']
    fp = o['fp']
    n = len(t)
    u = o['u']
    c.shape = (idim, n - k - 1)
    tcku = ([t, list(c), k], u)
    if ier <= 0 and (not quiet):
        warnings.warn(RuntimeWarning(_iermess[ier][0] + '\tk=%d n=%d m=%d fp=%f s=%f' % (k, len(t), m, fp, s)), stacklevel=2)
    if ier > 0 and (not full_output):
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]), stacklevel=2)
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return (tcku, fp, ier, _iermess[ier][0])
        except KeyError:
            return (tcku, fp, ier, _iermess['unknown'][0])
    else:
        return tcku