from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def ensure_single_select(*l):
    if not self.multiselect and len(self.selected_nodes) > 1:
        self.clear_selection()