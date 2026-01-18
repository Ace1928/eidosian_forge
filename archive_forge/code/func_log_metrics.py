from typing import Any, Dict, List, Optional
import torch
from composer.core.state import State
from composer.loggers import Logger
from composer.loggers.logger_destination import LoggerDestination
import ray.train
def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]=None) -> None:
    self.data.update(metrics.items())
    for key, val in self.data.items():
        if isinstance(val, torch.Tensor):
            self.data[key] = val.item()