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
def create_json_panel(self, title, config, filename=None, data=None):
    """Create new :class:`SettingsPanel`.

        .. versionadded:: 1.5.0

        Check the documentation of :meth:`add_json_panel` for more information.
        """
    if filename is None and data is None:
        raise Exception('You must specify either the filename or data')
    if filename is not None:
        with open(filename, 'r') as fd:
            data = json.loads(fd.read())
    else:
        data = json.loads(data)
    if not isinstance(data, list):
        raise ValueError('The first element must be a list')
    panel = SettingsPanel(title=title, settings=self, config=config)
    for setting in data:
        if 'type' not in setting:
            raise ValueError('One setting are missing the "type" element')
        ttype = setting['type']
        cls = self._types.get(ttype)
        if cls is None:
            raise ValueError('No class registered to handle the <%s> type' % setting['type'])
        del setting['type']
        str_settings = {}
        for key, item in setting.items():
            str_settings[str(key)] = item
        instance = cls(panel=panel, **str_settings)
        panel.add_widget(instance)
    return panel