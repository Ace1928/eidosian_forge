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
def set_def_tab(self, new_tab):
    if not issubclass(new_tab.__class__, TabbedPanelHeader):
        raise TabbedPanelException('`default_tab_class` should be                subclassed from `TabbedPanelHeader`')
    if self._default_tab == new_tab:
        return
    oltab = self._default_tab
    self._default_tab = new_tab
    self.remove_widget(oltab)
    self._original_tab = None
    self.switch_to(new_tab)
    new_tab.state = 'down'