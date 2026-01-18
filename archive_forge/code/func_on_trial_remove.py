from typing import Dict, Optional, TYPE_CHECKING
from ray.air._internal.usage import tag_scheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI
def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
    pass