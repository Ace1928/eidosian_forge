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
def remove_old_screen(transition):
    if old_current in self.children:
        self.remove_widget(old_current)
        self.transition = old_transition
    transition.unbind(on_complete=remove_old_screen)