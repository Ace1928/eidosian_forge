import unittest
import textwrap
from collections import defaultdict
def test_instantiate_from_kv(self):
    from kivy.lang import Builder

    class TestEventsFromKV(TrackCallbacks.get_base_class()):
        instantiated_widgets = []
    widget = Builder.load_string('TestEventsFromKV')
    self.assertIsInstance(widget, TestEventsFromKV)
    widget.root_widget = widget
    widget.base_widget = widget
    widget.check(self)