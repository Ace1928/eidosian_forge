from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
def hash_count(tokens: List[str]) -> Counter:
    hashed_tokens = [simple_hash(token, self.num_features) for token in tokens]
    return Counter(hashed_tokens)