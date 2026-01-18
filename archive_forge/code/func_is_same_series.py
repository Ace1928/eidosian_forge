import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
def is_same_series(self, other):
    """Return True if `other` appears to be in same series

        Parameters
        ----------
        other : object
           object with ``series_signature`` attribute that is a
           mapping.  Usually it's a ``Wrapper`` or sub-class instance.

        Returns
        -------
        tf : bool
           True if `other` might be in the same series as `self`, False
           otherwise.
        """
    my_sig = self.series_signature
    your_sig = other.series_signature
    my_keys = set(my_sig)
    your_keys = set(your_sig)
    for key in my_keys.intersection(your_keys):
        v1, func = my_sig[key]
        v2, _ = your_sig[key]
        if not func(v1, v2):
            return False
    for keys, sig in ((my_keys - your_keys, my_sig), (your_keys - my_keys, your_sig)):
        for key in keys:
            v1, func = sig[key]
            if not func(v1, None):
                return False
    return True