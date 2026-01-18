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
def next_tune_dataset(self, best_n: int=0) -> TuneDataset:
    """Convert the result back to a new :class:`~.TuneDataset` to be
        used by the next steps.

        :param best_n: top n result to extract, defaults to 0 (entire result)
        :return: a new dataset for tuning
        """
    data = self.result(best_n).drop([TUNE_REPORT_ID, TUNE_REPORT_METRIC, TUNE_REPORT], if_exists=True)
    return TuneDataset(data, dfs=self._dataset.dfs, keys=self._dataset.keys)