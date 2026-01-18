import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def feature_names_out(transformer, input_features):
    n = transformer.kw_args['n']
    return list(input_features) + [f'rnd{i}' for i in range(n)]