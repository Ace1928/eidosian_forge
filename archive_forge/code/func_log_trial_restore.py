import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def log_trial_restore(self, trial: 'Trial'):
    for logger_class, trial_loggers in self._class_trial_loggers.items():
        if trial in trial_loggers:
            trial_loggers[trial].flush()