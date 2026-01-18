import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def get_process_log_level(self):
    """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """
    log_level = trainer_log_levels[self.log_level]
    log_level_replica = trainer_log_levels[self.log_level_replica]
    log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
    log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
    return log_level_main_node if self.should_log else log_level_replica_node