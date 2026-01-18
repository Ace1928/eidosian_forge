from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def test_2_switch(self, *args):
    self._win = EventLoop.window
    self.clean_garbage()
    root = Builder.load_string(KV)
    self.render(root)
    self.assertLess(len(self._win.children), 2)
    group2 = root.ids.group2
    group1 = root.ids.group1
    self.move_frames(5)
    self.check_dropdown(present=False)
    self.assertFalse(group2.is_open)
    self.assertFalse(group1.is_open)
    TouchPoint(*group2.center)
    self.check_dropdown(present=True)
    g2dd = WeakProxy(self._win.children[0])
    self.assertIn(g2dd, self._win.children)
    self.assertEqual(g2dd, self._win.children[0])
    self.assertTrue(group2.is_open)
    self.assertFalse(group1.is_open)
    TouchPoint(0, 0)
    sleep(g2dd.min_state_time)
    self.move_frames(1)
    TouchPoint(*group1.center)
    sleep(g2dd.min_state_time)
    self.move_frames(1)
    self.assertNotEqual(g2dd, self._win.children[0])
    self.assertFalse(group2.is_open)
    self.assertTrue(group1.is_open)
    self.check_dropdown(present=True)
    TouchPoint(0, 0)
    sleep(g2dd.min_state_time)
    self.move_frames(1)
    self.check_dropdown(present=False)
    self.assertFalse(group2.is_open)
    self.assertFalse(group1.is_open)
    self.assertNotIn(g2dd, self._win.children)
    self._win.remove_widget(root)