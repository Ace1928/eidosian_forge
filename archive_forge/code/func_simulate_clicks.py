import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
def simulate_clicks(cv_data: RelDataCV) -> Tuple[ClickFold, Optional[ClickFold]]:
    """Simulate click data using position biased model (PBM)."""
    X, y, qid = list(zip(cv_data.train, cv_data.test))
    indptr = np.array([0] + [v.shape[0] for v in X])
    indptr = np.cumsum(indptr)
    assert len(indptr) == 2 + 1
    X_full = sparse.vstack(X)
    y_full = np.concatenate(y)
    qid_full = np.concatenate(qid)
    scores_full = init_rank_score(X_full, y_full, qid_full)
    scores = [scores_full[indptr[i - 1]:indptr[i]] for i in range(1, indptr.size)]
    X_lst, y_lst, q_lst, s_lst, c_lst, p_lst = ([], [], [], [], [], [])
    for i in range(indptr.size - 1):
        fold = simulate_one_fold((X[i], y[i], qid[i]), scores[i])
        X_lst.append(fold.X)
        y_lst.append(fold.y)
        q_lst.append(fold.qid)
        s_lst.append(fold.score)
        c_lst.append(fold.click)
        p_lst.append(fold.pos)
    scores_check_1 = [s_lst[i] for i in range(indptr.size - 1)]
    for i in range(2):
        assert (scores_check_1[i] == scores[i]).all()
    if len(X_lst) == 1:
        train = ClickFold(X_lst[0], y_lst[0], q_lst[0], s_lst[0], c_lst[0], p_lst[0])
        test = None
    else:
        train, test = (ClickFold(X_lst[i], y_lst[i], q_lst[i], s_lst[i], c_lst[i], p_lst[i]) for i in range(len(X_lst)))
    return (train, test)