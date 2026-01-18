import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
import cloudpickle
import numpy as np
import pandas as pd
from fugue import (
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
from tune.exceptions import TuneCompileError
def save_single_file(e: ExecutionEngine, _input: DataFrame) -> DataFrame:
    p = _get_temp_path(path, e.conf)
    fp = os.path.join(p, str(uuid4()) + '.parquet')
    e.save_df(_input, fp, force_single=True)
    return ArrayDataFrame([[fp]], f'{TUNE_DATASET_DF_PREFIX}{name}:str')