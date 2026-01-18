from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def bin_values(s: pd.Series) -> pd.Series:
    if s.name not in self.columns:
        return s
    labels = self.dtypes.get(s.name) if self.dtypes else False
    ordered = True
    if labels:
        if isinstance(labels, pd.CategoricalDtype):
            ordered = labels.ordered
            labels = list(labels.categories)
        else:
            labels = False
    bins = self.stats_ if self._is_fittable else self.bins
    return pd.cut(s, bins[s.name] if isinstance(bins, dict) else bins, right=self.right, labels=labels, ordered=ordered, retbins=False, include_lowest=self.include_lowest, duplicates=self.duplicates)