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
def add_json_panel(self, title, config, filename=None, data=None):
    """Create and add a new :class:`SettingsPanel` using the configuration
        `config` with the JSON definition `filename`. If `filename` is not set,
        then the JSON definition is read from the `data` parameter instead.

        Check the :ref:`settings_json` section in the documentation for more
        information about JSON format and the usage of this function.
        """
    panel = self.create_json_panel(title, config, filename, data)
    uid = panel.uid
    if self.interface is not None:
        self.interface.add_panel(panel, title, uid)