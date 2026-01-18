from typing import Any, Dict, List, Optional
import torch
from composer.core.state import State
from composer.loggers import Logger
from composer.loggers.logger_destination import LoggerDestination
import ray.train
def fit_end(self, state: State, logger: Logger) -> None:
    del logger
    if self.should_report_fit_end:
        ray.train.report(self.data)