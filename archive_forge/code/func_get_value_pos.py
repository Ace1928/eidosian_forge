from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
def get_value_pos(self):
    padding = self.padding
    x = self.x
    y = self.y
    nval = self.value_normalized
    if self.orientation == 'horizontal':
        return (x + padding + nval * (self.width - 2 * padding), y)
    else:
        return (x, y + padding + nval * (self.height - 2 * padding))