from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
def set_fs(self, value):
    """Attempt to set the fragment shader to the given value.
        If setting the shader fails, the existing one is preserved and an
        exception is raised.
        """
    shader = self.shader
    old_value = shader.fs
    shader.fs = value
    if not shader.success:
        shader.fs = old_value
        raise Exception('Setting new shader failed.')