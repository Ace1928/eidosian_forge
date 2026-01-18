import copy
from collections import defaultdict
import numpy as np
from pandas import compat, DataFrame
def nested_to_record(ds, prefix='', sep='.', level=0):
    """a simplified json_normalize

    converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : string, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

        .. versionadded:: 0.20.0

    level: the number of levels in the jason string, optional, default: 0

    Returns
    -------
    d - dict or list of dicts, matching `ds`

    Examples
    --------

    IN[52]: nested_to_record(dict(flat1=1,dict1=dict(c=1,d=2),
                                  nested=dict(e=dict(c=1,d=2),d=2)))
    Out[52]:
    {'dict1.c': 1,
     'dict1.d': 2,
     'flat1': 1,
     'nested.d': 2,
     'nested.e.c': 1,
     'nested.e.d': 2}
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
            if not isinstance(k, compat.string_types):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k
            if not isinstance(v, dict):
                if level != 0:
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue
            else:
                v = new_d.pop(k)
                new_d.update(nested_to_record(v, newkey, sep, level + 1))
        new_ds.append(new_d)
    if singleton:
        return new_ds[0]
    return new_ds