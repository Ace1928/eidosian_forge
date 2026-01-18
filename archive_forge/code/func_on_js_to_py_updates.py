import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
def on_js_to_py_updates(change):
    if self.debug:
        updates_str = json.dumps(change['new'], indent=2)
        print(f'JavaScript to Python VegaFusion updates:\n {updates_str}')
    updates = self._chart_state.update(change['new'])
    if self.debug:
        updates_str = json.dumps(updates, indent=2)
        print(f'Python to JavaScript VegaFusion updates:\n {updates_str}')
    self._py_to_js_updates = updates