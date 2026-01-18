import base64
import json
import logging
import os
from collections import namedtuple
from typing import (
import numpy as np
import pandas as pd
from pyspark import RDD, SparkContext, cloudpickle
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
from pyspark.ml.util import (
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, countDistinct, pandas_udf, rand, struct
from pyspark.sql.types import (
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
import xgboost
from xgboost import XGBClassifier
from xgboost.compat import is_cudf_available, is_cupy_available
from xgboost.core import Booster, _check_distributed_params
from xgboost.sklearn import DEFAULT_N_ESTIMATORS, XGBModel, _can_use_qdm
from xgboost.training import train as worker_train
from .._typing import ArrayLike
from .data import (
from .params import (
from .utils import (
@pandas_udf(schema)
def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    assert xgb_sklearn_model is not None
    model = xgb_sklearn_model
    from pyspark import TaskContext
    context = TaskContext.get()
    assert context is not None
    dev_ordinal = -1
    if is_cudf_available():
        if is_local:
            if run_on_gpu and is_cupy_available():
                import cupy as cp
                total_gpus = cp.cuda.runtime.getDeviceCount()
                if total_gpus > 0:
                    partition_id = context.partitionId()
                    dev_ordinal = partition_id % total_gpus
        elif run_on_gpu:
            dev_ordinal = _get_gpu_id(context)
        if dev_ordinal >= 0:
            device = 'cuda:' + str(dev_ordinal)
            get_logger('XGBoost-PySpark').info('Do the inference with device: %s', device)
            model.set_params(device=device)
        else:
            get_logger('XGBoost-PySpark').info('Do the inference on the CPUs')
    else:
        msg = 'CUDF is unavailable, fallback the inference on the CPUs' if run_on_gpu else 'Do the inference on the CPUs'
        get_logger('XGBoost-PySpark').info(msg)

    def to_gpu_if_possible(data: ArrayLike) -> ArrayLike:
        """Move the data to gpu if possible"""
        if dev_ordinal >= 0:
            import cudf
            import cupy as cp
            cp.cuda.runtime.setDevice(dev_ordinal)
            df = cudf.DataFrame(data)
            del data
            return df
        return data
    for data in iterator:
        if enable_sparse_data_optim:
            X = _read_csr_matrix_from_unwrapped_spark_vec(data)
        else:
            if feature_col_names is not None:
                tmp = data[feature_col_names]
            else:
                tmp = stack_series(data[alias.data])
            X = to_gpu_if_possible(tmp)
        if has_base_margin:
            base_margin = to_gpu_if_possible(data[alias.margin])
        else:
            base_margin = None
        yield predict_func(model, X, base_margin)