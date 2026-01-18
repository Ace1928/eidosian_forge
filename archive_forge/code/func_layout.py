import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
@layout.setter
def layout(self, d: Dict[str, int]) -> None:
    d['x'] = coalesce(d.get('x'), self._default_panel_layout()['x'])
    d['y'] = coalesce(d.get('y'), self._default_panel_layout()['y'])
    d['w'] = coalesce(d.get('w'), self._default_panel_layout()['w'])
    d['h'] = coalesce(d.get('h'), self._default_panel_layout()['h'])
    json_path = 'spec.layout'
    nested_set(self, json_path, d)