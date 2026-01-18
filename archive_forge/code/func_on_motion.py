from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def on_motion(self, etype, me):
    """Event called when a motion event is received.

        :Parameters:
            `etype`: str
                One of "begin", "update" or "end".
            `me`: :class:`~kivy.input.motionevent.MotionEvent`
                The motion event currently dispatched.

        .. versionchanged:: 2.1.0
            Event managers get to handle the touch event first and if none of
            them accepts the event (by returning `True`) then window will
            dispatch `me` through "on_touch_down", "on_touch_move",
            "on_touch_up" events depending on the `etype`. All non-touch events
            will go only through managers.
        """
    accepted = False
    for manager in self.event_managers_dict[me.type_id][:]:
        accepted = manager.dispatch(etype, me) or accepted
    if accepted:
        if me.is_touch and etype == 'end':
            FocusBehavior._handle_post_on_touch_up(me)
        return accepted
    if me.is_touch:
        self.transform_motion_event_2d(me)
        if etype == 'begin':
            self.dispatch('on_touch_down', me)
        elif etype == 'update':
            self.dispatch('on_touch_move', me)
        elif etype == 'end':
            self.dispatch('on_touch_up', me)
            FocusBehavior._handle_post_on_touch_up(me)