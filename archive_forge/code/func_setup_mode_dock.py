from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def setup_mode_dock(self, *largs):
    """Setup the keyboard in docked mode.

        Dock mode will reset the rotation, disable translation, rotation and
        scale. Scale and position will be automatically adjusted to attach the
        keyboard to the bottom of the screen.

        .. note::
            Don't call this method directly, use :meth:`setup_mode` instead.
        """
    self.do_translation = False
    self.do_rotation = False
    self.do_scale = False
    self.rotation = 0
    win = self.get_parent_window()
    scale = win.width / float(self.width)
    self.scale = scale
    self.pos = (0, 0)
    win.bind(on_resize=self._update_dock_mode)