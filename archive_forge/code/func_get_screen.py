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
def get_screen(self, name):
    """Return the screen widget associated with the name or raise a
        :class:`ScreenManagerException` if not found.
        """
    matches = [s for s in self.screens if s.name == name]
    num_matches = len(matches)
    if num_matches == 0:
        raise ScreenManagerException('No Screen with name "%s".' % name)
    if num_matches > 1:
        Logger.warn('Multiple screens named "%s": %s' % (name, matches))
    return matches[0]