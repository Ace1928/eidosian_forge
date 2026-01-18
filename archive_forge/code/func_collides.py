import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def collides(p1: Panel, p2: Panel) -> bool:
    l1, l2 = (p1.layout, p2.layout)
    if p1.spec['__id__'] == p2.spec['__id__'] or l1['x'] + l1['w'] <= l2['x'] or l1['x'] >= l2['w'] + l2['x'] or (l1['y'] + l1['h'] <= l2['y']) or (l1['y'] >= l2['y'] + l2['h']):
        return False
    return True