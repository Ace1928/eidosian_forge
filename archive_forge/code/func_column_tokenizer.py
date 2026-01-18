from typing import Callable, List, Optional
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_split_tokenizer
from ray.util.annotations import PublicAPI
def column_tokenizer(s: pd.Series):
    return s.map(self.tokenization_fn)