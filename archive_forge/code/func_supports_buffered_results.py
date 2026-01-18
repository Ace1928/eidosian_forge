from typing import Dict, Optional, TYPE_CHECKING
from ray.air._internal.usage import tag_scheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI
@property
def supports_buffered_results(self):
    return self._supports_buffered_results