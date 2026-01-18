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
def add_panel(self, name, cb_activate, cb_deactivate, cb_refresh=None):
    """Add a new panel in the Console.

        - `cb_activate` is a callable that will be called when the panel is
          activated by the user.

        - `cb_deactivate` is a callable that will be called when the panel is
          deactivated or when the console will hide.

        - `cb_refresh` is an optional callable that is called if the user
          click again on the button for display the panel

        When activated, it's up to the panel to display a content in the
        Console by using :meth:`set_content`.
        """
    btn = ConsoleToggleButton(text=name)
    btn.cb_activate = cb_activate
    btn.cb_deactivate = cb_deactivate
    btn.cb_refresh = cb_refresh
    btn.bind(on_press=self._activate_panel)
    self._toolbar['panels'].append(btn)