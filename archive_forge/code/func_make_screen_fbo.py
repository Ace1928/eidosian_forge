from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
def make_screen_fbo(self, screen):
    fbo = Fbo(size=screen.size, with_stencilbuffer=True)
    with fbo:
        ClearColor(*self.clearcolor)
        ClearBuffers()
    fbo.add(screen.canvas)
    with fbo.before:
        PushMatrix()
        Translate(-screen.x, -screen.y, 0)
    with fbo.after:
        PopMatrix()
    return fbo