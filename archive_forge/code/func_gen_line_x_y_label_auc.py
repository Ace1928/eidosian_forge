import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def gen_line_x_y_label_auc(_y, _y_prob, _pos_label):
    precision, recall, _ = sk_metrics.precision_recall_curve(_y, _y_prob, sample_weight=sample_weights, pos_label=_pos_label if _pos_label == pos_label else None)
    ap = sk_metrics.average_precision_score(y_true=_y, y_score=_y_prob, pos_label=_pos_label, sample_weight=sample_weights)
    return (recall, precision, f'AP={ap:.3f}', ap)