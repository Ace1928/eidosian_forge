import json
import os
import kivy.utils as utils
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.config import ConfigParser
from kivy.animation import Animation
from kivy.compat import string_types, text_type
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty, ListProperty, \
def add_interface(self):
    """(Internal) creates an instance of :attr:`Settings.interface_cls`,
        and sets it to :attr:`~Settings.interface`. When json panels are
        created, they will be added to this interface which will display them
        to the user.
        """
    cls = self.interface_cls
    if isinstance(cls, string_types):
        cls = Factory.get(cls)
    interface = cls()
    self.interface = interface
    self.add_widget(interface)
    self.interface.bind(on_close=lambda j: self.dispatch('on_close'))