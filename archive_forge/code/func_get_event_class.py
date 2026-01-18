import pytest
from string import ascii_letters
from random import randint
import gc
import sys
def get_event_class(name, args, kwargs):
    from kivy.event import EventDispatcher
    import kivy.properties
    from kivy.properties import BooleanProperty, ReferenceListProperty, AliasProperty
    if name == 'AliasProperty':

        class Event(EventDispatcher):

            def get_a(self):
                return 0

            def set_a(self, value):
                pass
            a = AliasProperty(get_a, set_a)
    elif name == 'ReferenceListProperty':

        class Event(EventDispatcher):
            a1 = BooleanProperty(0)
            a2 = BooleanProperty(0)
            a = ReferenceListProperty(a1, a2)
    else:
        cls = getattr(kivy.properties, name)

        class Event(EventDispatcher):
            a = cls(*args, **kwargs)
    return Event