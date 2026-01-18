import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def plot_lines(data_series, xlabel, ylabel, legend_loc=None, line_kwargs=None, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if line_kwargs is None:
        line_kwargs = {}
    for label, data_x, data_y in data_series:
        ax.plot(data_x, data_y, label=label, **line_kwargs)
    if legend_loc:
        ax.legend(loc=legend_loc)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return (fig, ax)