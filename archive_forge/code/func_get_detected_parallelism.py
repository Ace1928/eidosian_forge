from typing import Any, Dict, Optional, Union
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.datasource.datasource import Datasource, Reader
def get_detected_parallelism(self) -> int:
    """
        Get the true parallelism that should be used during execution.
        """
    return self._detected_parallelism