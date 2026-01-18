from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def check_dropdown(self, present=True):
    any_list = [isinstance(child, DropDown) for child in self._win.children]
    self.assertLess(sum(any_list), 2)
    if not present and (not any(any_list)):
        return
    elif present and any(any_list):
        return
    print("DropDown either missing, or isn't supposed to be there")
    self.assertTrue(False)