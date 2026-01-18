import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def on_anytrait_changed(self, object, trait_name, old, new):
    if trait_name == 'value1':
        self.obj.value1_count += 1
    elif trait_name == 'value2':
        self.obj.value2_count += 1