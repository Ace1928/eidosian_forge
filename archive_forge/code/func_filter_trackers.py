import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
def filter_trackers(log_with: List[Union[str, LoggerType, GeneralTracker]], logging_dir: Union[str, os.PathLike]=None):
    """
    Takes in a list of potential tracker types and checks that:
        - The tracker wanted is available in that environment
        - Filters out repeats of tracker types
        - If `all` is in `log_with`, will return all trackers in the environment
        - If a tracker requires a `logging_dir`, ensures that `logging_dir` is not `None`

    Args:
        log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
            A list of loggers to be setup for experiment tracking. Should be one or several of:

            - `"all"`
            - `"tensorboard"`
            - `"wandb"`
            - `"comet_ml"`
            - `"mlflow"`
            - `"dvclive"`
            If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
            also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        logging_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing logs of locally-compatible loggers.
    """
    loggers = []
    if log_with is not None:
        if not isinstance(log_with, (list, tuple)):
            log_with = [log_with]
        if 'all' in log_with or LoggerType.ALL in log_with:
            loggers = [o for o in log_with if issubclass(type(o), GeneralTracker)] + get_available_trackers()
        else:
            for log_type in log_with:
                if log_type not in LoggerType and (not issubclass(type(log_type), GeneralTracker)):
                    raise ValueError(f'Unsupported logging capability: {log_type}. Choose between {LoggerType.list()}')
                if issubclass(type(log_type), GeneralTracker):
                    loggers.append(log_type)
                else:
                    log_type = LoggerType(log_type)
                    if log_type not in loggers:
                        if log_type in get_available_trackers():
                            tracker_init = LOGGER_TYPE_TO_CLASS[str(log_type)]
                            if tracker_init.requires_logging_directory:
                                if logging_dir is None:
                                    raise ValueError(f'Logging with `{log_type}` requires a `logging_dir` to be passed in.')
                            loggers.append(log_type)
                        else:
                            logger.debug(f'Tried adding logger {log_type}, but package is unavailable in the system.')
    return loggers