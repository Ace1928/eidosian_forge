from functools import partial
from kivy.clock import Clock
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, OptionProperty, \
def on_do_default_tab(self, instance, value):
    if not value:
        dft = self.default_tab
        if dft in self.tab_list:
            self.remove_widget(dft)
            self._switch_to_first_tab()
            self._default_tab = self._current_tab
    else:
        self._current_tab.state = 'normal'
        self._setup_default_tab()