import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def on_toggle(self, value: bool):
    """Event: returns True or False to indicate the current state."""