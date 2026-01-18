import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.fixture(autouse=True)
def set_clock(kivy_clock):
    pass