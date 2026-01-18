from typing import List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import StatsDict
def close_sub_progress_bars(self):
    """Close all internal sub progress bars."""
    if self._sub_progress_bar_dict is not None:
        for sub_bar in self._sub_progress_bar_dict.values():
            sub_bar.close()