import kivy
import weakref
from functools import partial
from itertools import chain
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.treeview import TreeViewNode, TreeView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix
from kivy.graphics.context_instructions import Transform
from kivy.graphics.transformation import Matrix
from kivy.properties import (ObjectProperty, BooleanProperty, ListProperty,
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
def highlight_at(self, x, y):
    """Select a widget from a x/y window coordinate.
        This is mostly used internally when Select mode is activated
        """
    widget = None
    win_children = self.win.children
    children = chain((c for c in reversed(win_children) if isinstance(c, ModalView)), (c for c in reversed(win_children) if not isinstance(c, ModalView)))
    for child in children:
        if child is self:
            continue
        widget = self.pick(child, x, y)
        if widget:
            break
    self.highlight_widget(widget)