import collections
from typing import List
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash
from ray.util.annotations import PublicAPI
def row_feature_hasher(row):
    hash_counts = collections.defaultdict(int)
    for column in self.columns:
        hashed_value = simple_hash(column, self.num_features)
        hash_counts[hashed_value] += row[column]
    return {f'hash_{i}': hash_counts[i] for i in range(self.num_features)}