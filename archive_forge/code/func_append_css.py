import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def append_css(self, stylesheet):
    self._resources.append_resource(stylesheet)