from typing import Optional
from ...public import PanelMetricsHelper, Run
from .runset import Runset
from .util import Attr, Base, Panel, nested_get, nested_set
@classmethod
def from_run(cls, run: 'Run', metric: str) -> 'LineKey':
    key = f'{run.id}:{metric}'
    return cls(key)