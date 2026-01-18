from typing import Dict, List, Optional
from ray.tune.search import Searcher, ConcurrencyLimiter
from ray.tune.search.search_generator import SearchGenerator
from ray.tune.experiment import Trial
@property
def live_trials(self) -> List[Trial]:
    return self.searcher.live_trials