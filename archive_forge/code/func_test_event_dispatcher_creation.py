import pytest
from string import ascii_letters
from random import randint
import gc
import sys
def test_event_dispatcher_creation(kivy_benchmark):
    from kivy.event import EventDispatcher

    class Event(EventDispatcher):
        pass
    e = Event()
    kivy_benchmark(Event)