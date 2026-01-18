from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
def get_reports(self, trial: Trial) -> List[TrialReport]:
    raise NotImplementedError