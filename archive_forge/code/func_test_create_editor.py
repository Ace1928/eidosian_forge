import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_create_editor(self):
    from traitsui.testing.api import UITester
    obj = EnumCollectionGUIExample()
    with UITester().create_ui(obj):
        pass