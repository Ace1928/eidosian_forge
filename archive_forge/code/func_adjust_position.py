from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import OptionProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.properties import ColorProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.base import EventLoop
from kivy.metrics import dp
def adjust_position(self):
    if self.limit_to is not None and (not self._temporarily_ignore_limits):
        if self.limit_to is EventLoop.window:
            lim_x, lim_y = (0, 0)
            lim_top, lim_right = self.limit_to.size
        else:
            lim_x = self.limit_to.x
            lim_y = self.limit_to.y
            lim_top = self.limit_to.top
            lim_right = self.limit_to.right
        self._temporarily_ignore_limits = True
        if not (lim_x > self.x and lim_right < self.right):
            self.x = max(lim_x, min(lim_right - self.width, self.x))
        if not (lim_y > self.y and lim_right < self.right):
            self.y = min(lim_top - self.height, max(lim_y, self.y))
        self._temporarily_ignore_limits = False