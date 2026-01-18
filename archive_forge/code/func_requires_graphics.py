import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def requires_graphics(func):
    if 'mock' == cgl_get_backend_name():
        return pytest.mark.skip(reason='Skipping because gl backend is set to mock')(func)
    return func