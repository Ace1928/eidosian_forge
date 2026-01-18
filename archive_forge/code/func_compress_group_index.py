from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def compress_group_index(group_index: npt.NDArray[np.int64], sort: bool=True) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    if len(group_index) and np.all(group_index[1:] >= group_index[:-1]):
        unique_mask = np.concatenate([group_index[:1] > -1, group_index[1:] != group_index[:-1]])
        comp_ids = unique_mask.cumsum()
        comp_ids -= 1
        obs_group_ids = group_index[unique_mask]
    else:
        size_hint = len(group_index)
        table = hashtable.Int64HashTable(size_hint)
        group_index = ensure_int64(group_index)
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)
        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)
    return (ensure_int64(comp_ids), ensure_int64(obs_group_ids))