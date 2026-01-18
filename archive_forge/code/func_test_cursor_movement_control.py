import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_cursor_movement_control(self):
    text = 'these are\nmany words'
    ti = TextInput(multiline=True, text=text)
    ti.focus = True
    self.render(ti)
    self.assertTrue(ti.focus)
    self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 1))
    options = (('cursor_left', (5, 1)), ('cursor_left', (0, 1)), ('cursor_left', (6, 0)), ('cursor_right', (9, 0)), ('cursor_right', (4, 1)))
    for key, pos in options:
        ti._key_down((None, None, 'ctrl_L', 1), repeat=False)
        ti._key_down((None, None, key, 1), repeat=False)
        self.assertEqual(ti.cursor, pos)
        ti._key_up((None, None, 'ctrl_L', 1), repeat=False)