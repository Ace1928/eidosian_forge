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
def setup_mode_free(self):
    """Setup the keyboard in free mode.

        Free mode is designed to let the user control the position and
        orientation of the keyboard. The only real usage is for a multiuser
        environment, but you might found other ways to use it.
        If a :attr:`target` is set, it will place the vkeyboard under the
        target.

        .. note::
            Don't call this method directly, use :meth:`setup_mode` instead.
        """
    self.do_translation = True
    self.do_rotation = True
    self.do_scale = True
    target = self.target
    if not target:
        return
    a = Vector(1, 0)
    b = Vector(target.to_window(0, 0))
    c = Vector(target.to_window(1, 0)) - b
    self.rotation = -a.angle(c)
    dpos = Vector(self.to_window(self.width / 2.0, self.height))
    cpos = Vector(target.to_window(target.center_x, target.y))
    diff = dpos - cpos
    diff2 = Vector(self.x + self.width / 2.0, self.y + self.height) - Vector(self.to_parent(self.width / 2.0, self.height))
    diff -= diff2
    self.pos = -diff