import unittest
from kivy.tests.common import GraphicUnitTest, UnitTestTouch
from kivy.base import EventLoop
from kivy.modules import inspector
from kivy.factory import Factory
def test_activate_deactivate_top(self, *args):
    EventLoop.ensure_window()
    self._win = EventLoop.window
    self.clean_garbage()
    self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
    self.render(self.root)
    self.assertLess(len(self._win.children), 2)
    inspector.start(self._win, self.root)
    self.advance_frames(2)
    ins = self.root.inspector
    ins.at_bottom = False
    ins.activated = True
    ins.inspect_enabled = True
    self.assertFalse(ins.at_bottom)
    self.assertEqual(self._win.children[0], ins)
    ins.toggle_position(self.root.ids.dummy)
    self.advance_frames(20)
    self.assertGreater(ins.layout.pos[1], self._win.height / 2.0)
    ins.inspect_enabled = False
    ins.activated = False
    self.render(self.root)
    self.advance_frames(1)
    inspector.stop(self._win, self.root)
    self.assertLess(len(self._win.children), 2)
    self.render(self.root)