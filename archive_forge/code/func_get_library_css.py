import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def get_library_css(self, libraries):
    return self._resources.get_library_resources(libraries)