from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def get_classification_hyperparams(df: pd.DataFrame) -> tuple[int, object]:
    n_classes = df.completion.nunique()
    pos_class = None
    if n_classes == 2:
        pos_class = df.completion.value_counts().index[0]
    return (n_classes, pos_class)