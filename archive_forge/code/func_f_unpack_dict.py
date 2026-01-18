import time
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively.
    There should be no duplicated keys across all nested
    subdictionaries, or some instances will be lost without warning

    Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """
    res = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v
    return res