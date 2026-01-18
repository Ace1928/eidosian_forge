from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
def set_norm_value(self, value):
    vmin = self.min
    vmax = self.max
    step = self.step
    val = min(value * (vmax - vmin) + vmin, vmax)
    if step == 0:
        self.value = val
    else:
        self.value = min(round((val - vmin) / step) * step + vmin, vmax)