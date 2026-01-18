import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
def replace_none_in_shape(shape):
    return [-1 if dim_size is None else dim_size for dim_size in shape]