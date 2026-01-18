from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
def ending_round(self, index: int, state: Any) -> None:
    logger.info('Reporter.ending_round(%r, state)', index)