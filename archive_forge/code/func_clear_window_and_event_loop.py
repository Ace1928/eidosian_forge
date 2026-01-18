import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def clear_window_and_event_loop(self):
    from kivy.base import EventLoop
    window = self.Window
    for child in window.children[:]:
        window.remove_widget(child)
    window.canvas.before.clear()
    window.canvas.clear()
    window.canvas.after.clear()
    EventLoop.touches.clear()
    for post_proc in EventLoop.postproc_modules:
        if hasattr(post_proc, 'touches'):
            post_proc.touches.clear()
        elif hasattr(post_proc, 'last_touches'):
            post_proc.last_touches.clear()