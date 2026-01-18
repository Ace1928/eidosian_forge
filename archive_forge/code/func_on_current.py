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
def on_current(self, instance, value):
    if value is None:
        self.transition.stop()
        self.current_screen = None
        return
    screen = self.get_screen(value)
    if screen == self.current_screen:
        return
    self.transition.stop()
    previous_screen = self.current_screen
    self.current_screen = screen
    if previous_screen:
        self.transition.screen_in = screen
        self.transition.screen_out = previous_screen
        self.transition.start(self)
    else:
        self.real_add_widget(screen)
        screen.pos = self.pos
        self.do_layout()
        screen.dispatch('on_pre_enter')
        screen.dispatch('on_enter')