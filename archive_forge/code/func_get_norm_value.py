from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
def get_norm_value(self):
    vmin = self.min
    d = self.max - vmin
    if d == 0:
        return 0
    return (self.value - vmin) / float(d)