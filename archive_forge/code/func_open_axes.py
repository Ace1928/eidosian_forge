import warnings
import json
import random
from .base import Renderer
from ..exporter import Exporter
def open_axes(self, ax, props):
    if len(self.axes) > 0:
        warnings.warn('multiple axes not yet supported')
    self.axes = [dict(type='x', scale='x', ticks=10), dict(type='y', scale='y', ticks=10)]
    self.scales = [dict(name='x', domain=props['xlim'], type='linear', range='width'), dict(name='y', domain=props['ylim'], type='linear', range='height')]