import logging
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.cards import pandas_renderer
from mlflow.utils.databricks_utils import (
from mlflow.utils.os import is_windows
def get_pandas_data_profiles(inputs: Iterable[Tuple[str, pd.DataFrame]]) -> str:
    """
    Returns a data profiling string over input data frame.

    Args:
        inputs: Either a single "glimpse" DataFrame that contains the statistics, or a
            collection of (title, DataFrame) pairs where each pair names a separate "glimpse"
            and they are all visualized in comparison mode.

    Returns:
        a data profiling string such as Pandas profiling ProfileReport.
    """
    truncated_input = [truncate_pandas_data_profile(*input) for input in inputs]
    return pandas_renderer.get_html(truncated_input)