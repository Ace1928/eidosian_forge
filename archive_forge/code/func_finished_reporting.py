import copy
import logging
from typing import Dict, List, Optional
import numpy as np
from ray.tune.search import Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util import PublicAPI
def finished_reporting(self) -> bool:
    return None not in self._trials.values() and len(self._trials) == self.max_trials