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
def show_property(self, instance, value, key=None, index=-1, *l):
    if value is False:
        return
    console = self.console
    content = None
    if key is None:
        nested = False
        widget = instance.widget
        key = instance.key
        prop = widget.property(key)
        value = getattr(widget, key)
    else:
        nested = True
        widget = instance
        prop = None
    dtype = None
    if isinstance(prop, AliasProperty) or nested:
        if type(value) in (str, str):
            dtype = 'string'
        elif type(value) in (int, float):
            dtype = 'numeric'
        elif type(value) in (tuple, list):
            dtype = 'list'
    if isinstance(prop, NumericProperty) or dtype == 'numeric':
        content = TextInput(text=str(value) or '', multiline=False)
        content.bind(text=partial(self.save_property_numeric, widget, key, index))
    elif isinstance(prop, StringProperty) or dtype == 'string':
        content = TextInput(text=value or '', multiline=True)
        content.bind(text=partial(self.save_property_text, widget, key, index))
    elif isinstance(prop, ListProperty) or isinstance(prop, ReferenceListProperty) or isinstance(prop, VariableListProperty) or (dtype == 'list'):
        content = GridLayout(cols=1, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        for i, item in enumerate(value):
            button = Button(text=repr(item), size_hint_y=None, height=44)
            if isinstance(item, Widget):
                button.bind(on_release=partial(console.highlight_widget, item, False))
            else:
                button.bind(on_release=partial(self.show_property, widget, item, key, i))
            content.add_widget(button)
    elif isinstance(prop, OptionProperty):
        content = GridLayout(cols=1, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        for option in prop.options:
            button = ToggleButton(text=option, state='down' if option == value else 'normal', group=repr(content.uid), size_hint_y=None, height=44)
            button.bind(on_press=partial(self.save_property_option, widget, key))
            content.add_widget(button)
    elif isinstance(prop, ObjectProperty):
        if isinstance(value, Widget):
            content = Button(text=repr(value))
            content.bind(on_release=partial(console.highlight_widget, value))
        elif isinstance(value, Texture):
            content = Image(texture=value)
        else:
            content = Label(text=repr(value))
    elif isinstance(prop, BooleanProperty):
        state = 'down' if value else 'normal'
        content = ToggleButton(text=key, state=state)
        content.bind(on_release=partial(self.save_property_boolean, widget, key, index))
    self.root.clear_widgets()
    self.root.add_widget(self.sv)
    if content:
        self.root.add_widget(content)