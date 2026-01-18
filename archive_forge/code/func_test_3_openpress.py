from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def test_3_openpress(self, *args):
    self._win = EventLoop.window
    self.clean_garbage()
    root = Builder.load_string(KV)
    self.render(root)
    self.assertLess(len(self._win.children), 2)
    group2 = root.ids.group2
    group2button = root.ids.group2button
    group1 = root.ids.group1
    group1button = root.ids.group1button
    self.move_frames(5)
    self.check_dropdown(present=False)
    self.assertFalse(group2.is_open)
    self.assertFalse(group1.is_open)
    items = ((group2, group1, group2button), (group1, group2, group1button))
    for item in items:
        active, passive, button = item
        TouchPoint(*active.center)
        self.check_dropdown(present=True)
        gdd = WeakProxy(self._win.children[0])
        self.assertIn(gdd, self._win.children)
        self.assertEqual(gdd, self._win.children[0])
        self.assertTrue(active.is_open)
        self.assertFalse(passive.is_open)
        TouchPoint(*button.to_window(*button.center))
        self.assertTrue(getattr(root, active.text[0::6] + 'button'))
        sleep(gdd.min_state_time)
        self.move_frames(1)
        self.assertNotEqual(gdd, self._win.children[0])
        self.assertLess(len(self._win.children), 2)
        self.assertFalse(active.is_open)
        self.assertFalse(passive.is_open)
        self.check_dropdown(present=False)
    self._win.remove_widget(root)