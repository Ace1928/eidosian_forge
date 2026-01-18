import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def post_dispatch_input(self, etype, me):
    """This function is called by :meth:`EventLoopBase.dispatch_input()`
        when we want to dispatch an input event. The event is dispatched to
        all listeners and if grabbed, it's dispatched to grabbed widgets.
        """
    if etype == 'begin':
        self.me_list.append(me)
    elif etype == 'end':
        if me in self.me_list:
            self.me_list.remove(me)
    if not me.grab_exclusive_class:
        for listener in self.event_listeners:
            listener.dispatch('on_motion', etype, me)
    if not me.is_touch:
        return
    me.grab_state = True
    for weak_widget in me.grab_list[:]:
        wid = weak_widget()
        if wid is None:
            me.grab_list.remove(weak_widget)
            continue
        root_window = wid.get_root_window()
        if wid != root_window and root_window is not None:
            me.push()
            try:
                root_window.transform_motion_event_2d(me, wid)
            except AttributeError:
                me.pop()
                continue
        me.grab_current = wid
        wid._context.push()
        if etype == 'begin':
            pass
        elif etype == 'update':
            if wid._context.sandbox:
                with wid._context.sandbox:
                    wid.dispatch('on_touch_move', me)
            else:
                wid.dispatch('on_touch_move', me)
        elif etype == 'end':
            if wid._context.sandbox:
                with wid._context.sandbox:
                    wid.dispatch('on_touch_up', me)
            else:
                wid.dispatch('on_touch_up', me)
        wid._context.pop()
        me.grab_current = None
        if wid != root_window and root_window is not None:
            me.pop()
    me.grab_state = False
    me.dispatch_done()