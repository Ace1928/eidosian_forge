from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
def get_pd_value_counts_per_column(col: pd.Series):
    if _is_series_composed_of_lists(col):
        if encode_lists:
            counter = Counter()

            def update_counter(element):
                counter.update(element)
                return element
            col.map(update_counter)
            return counter
        else:
            col = col.map(lambda x: tuple(x))
    return Counter(col.value_counts(dropna=False).to_dict())