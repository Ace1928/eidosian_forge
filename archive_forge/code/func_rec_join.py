import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_rec_join_dispatcher)
def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2', defaults=None):
    """
    Join arrays `r1` and `r2` on keys.
    Alternative to join_by, that always returns a np.recarray.

    See Also
    --------
    join_by : equivalent function
    """
    kwargs = dict(jointype=jointype, r1postfix=r1postfix, r2postfix=r2postfix, defaults=defaults, usemask=False, asrecarray=True)
    return join_by(key, r1, r2, **kwargs)