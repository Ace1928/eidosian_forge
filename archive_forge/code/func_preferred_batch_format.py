from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Union
import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def preferred_batch_format(cls) -> BatchFormat:
    return BatchFormat.NUMPY