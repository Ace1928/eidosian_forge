import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def per_substance_varied(self, per_substance, varied=None):
    """Dense nd-array for all combinations of varied levels per substance

        Parameters
        ----------
        per_substance: dict or array
        varied: dict

        Examples
        --------
        >>> rsys = ReactionSystem([], 'A B C')
        >>> arr, keys = rsys.per_substance_varied({'A': 2, 'B': 3, 'C': 5}, {'C': [5, 7, 9, 11]})
        >>> arr.shape, keys
        ((4, 3), ('C',))
        >>> all(arr[1, :] == [2, 3, 7])
        True

        Returns
        -------
        ndarray : with len(varied) + 1 number of axes, and with last axis length == self.ns

        """
    import numpy as np
    varied = varied or {}
    varied_keys = tuple((k for k in self.substances if k in varied))
    n_varied = len(varied)
    shape = tuple((len(varied[k]) for k in self.substances if k in varied))
    result = np.empty(shape + (self.ns,))
    result[..., :] = self.as_per_substance_array(per_substance)
    if varied:
        for k, vals in varied.items():
            varied_axis = varied_keys.index(k)
            for varied_idx, val in enumerate(vals):
                index = tuple((varied_idx if i == varied_axis else slice(None) for i in range(n_varied)))
                result[index + (self.as_substance_index(k),)] = val
    return (result, varied_keys)