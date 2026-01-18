from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
def compute_and_log(self, step: Optional[int]=None, **kwargs: Any) -> _THROUGHPUT_METRICS:
    """See :meth:`Throughput.compute`

        Args:
            step: Can be used to override the logging step.
            \\**kwargs: See available parameters in :meth:`Throughput.compute`

        """
    self.step = self.step + 1 if step is None else step
    metrics = self.compute(**kwargs)
    self._fabric.log_dict(metrics=metrics, step=self.step)
    return metrics