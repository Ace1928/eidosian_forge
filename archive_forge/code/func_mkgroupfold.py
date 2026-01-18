import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
def mkgroupfold(dall: DMatrix, nfold: int, param: BoosterParam, evals: Sequence[str]=(), fpreproc: Optional[FPreProcCallable]=None, shuffle: bool=True) -> List[CVPack]:
    """
    Make n folds for cross-validation maintaining groups
    :return: cross-validation folds
    """
    group_boundaries = dall.get_uint_info('group_ptr')
    group_sizes = np.diff(group_boundaries)
    if shuffle is True:
        idx = np.random.permutation(len(group_sizes))
    else:
        idx = np.arange(len(group_sizes))
    out_group_idset = np.array_split(idx, nfold)
    in_group_idset = [np.concatenate([out_group_idset[i] for i in range(nfold) if k != i]) for k in range(nfold)]
    in_idset = [groups_to_rows(in_groups, group_boundaries) for in_groups in in_group_idset]
    out_idset = [groups_to_rows(out_groups, group_boundaries) for out_groups in out_group_idset]
    ret = []
    for k in range(nfold):
        dtrain = dall.slice(in_idset[k], allow_groups=True)
        dtrain.set_group(group_sizes[in_group_idset[k]])
        dtest = dall.slice(out_idset[k], allow_groups=True)
        dtest.set_group(group_sizes[out_group_idset[k]])
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret