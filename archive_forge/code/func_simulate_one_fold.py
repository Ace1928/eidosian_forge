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
def simulate_one_fold(fold: Tuple[sparse.csr_matrix, npt.NDArray[np.int32], npt.NDArray[np.int32]], scores_fold: npt.NDArray[np.float32]) -> ClickFold:
    """Simulate clicks for one fold."""
    X_fold, y_fold, qid_fold = fold
    assert qid_fold.dtype == np.int32
    qids = np.unique(qid_fold)
    position = np.empty((y_fold.size,), dtype=np.int64)
    clicks = np.empty((y_fold.size,), dtype=np.int32)
    pbm = PBM(eta=1.0)
    for q in qids:
        qid_mask = q == qid_fold
        qid_mask = qid_mask.reshape(qid_mask.shape[0])
        query_scores = scores_fold[qid_mask]
        query_position = np.argsort(query_scores)[::-1]
        position[qid_mask] = query_position
        relevance_degrees = y_fold[qid_mask]
        query_clicks = pbm.sample_clicks_for_query(relevance_degrees, query_position)
        clicks[qid_mask] = query_clicks
    assert X_fold.shape[0] == qid_fold.shape[0], (X_fold.shape, qid_fold.shape)
    assert X_fold.shape[0] == clicks.shape[0], (X_fold.shape, clicks.shape)
    return ClickFold(X_fold, y_fold, qid_fold, scores_fold, clicks, position)