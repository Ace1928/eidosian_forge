from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
def get_token_counts(col):
    token_series = df[col].apply(self.tokenization_fn)
    tokens = token_series.sum()
    return Counter(tokens)