import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_cursor_blink(self):
    ti = TextInput(cursor_blink=True)
    ti.focus = True
    ti._do_blink_cursor_ev = Clock.create_trigger(ti._do_blink_cursor, 0.01, interval=True)
    self.render(ti)
    self.assertTrue(ti.cursor_blink)
    self.assertTrue(ti._do_blink_cursor_ev.is_triggered)
    ti.cursor_blink = False
    for i in range(30):
        self.advance_frames(int(0.01 * Clock._max_fps) + 1)
        self.assertFalse(ti._do_blink_cursor_ev.is_triggered)
        self.assertFalse(ti._cursor_blink)
    ti.cursor_blink = True
    self.assertTrue(ti.cursor_blink)
    for i in range(30):
        self.advance_frames(int(0.01 * Clock._max_fps) + 1)
        self.assertTrue(ti._do_blink_cursor_ev.is_triggered)