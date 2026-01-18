from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
def on_scroll_move(self, touch):
    if self._get_uid('svavoid') in touch.ud:
        return False
    touch.push()
    touch.apply_transform_2d(self.to_local)
    if self.dispatch_children('on_scroll_move', touch):
        touch.pop()
        return True
    touch.pop()
    rv = True
    touch.ud['sv.can_defocus'] = True
    uid = self._get_uid()
    if uid not in touch.ud:
        self._touch = False
        return self.on_scroll_start(touch, False)
    ud = touch.ud[uid]
    if ud['mode'] == 'unknown':
        if not (self.do_scroll_x or self.do_scroll_y):
            touch.push()
            touch.apply_transform_2d(self.to_local)
            touch.apply_transform_2d(self.to_window)
            self._change_touch_mode()
            touch.pop()
            return
        ud['dx'] += abs(touch.dx)
        ud['dy'] += abs(touch.dy)
        if ud['dx'] > self.scroll_distance and self.do_scroll_x or (ud['dy'] > self.scroll_distance and self.do_scroll_y):
            ud['mode'] = 'scroll'
    if ud['mode'] == 'scroll':
        not_in_bar = not touch.ud.get('in_bar_x', False) and (not touch.ud.get('in_bar_y', False))
        if not touch.ud['sv.handled']['x'] and self.do_scroll_x and self.effect_x:
            width = self.width
            if touch.ud.get('in_bar_x', False):
                if self.hbar[1] != 1:
                    dx = touch.dx / float(width - width * self.hbar[1])
                    self.scroll_x = min(max(self.scroll_x + dx, 0.0), 1.0)
                    self._trigger_update_from_scroll()
            elif not_in_bar:
                self.effect_x.update(touch.x)
            if self.scroll_x < 0 or self.scroll_x > 1:
                rv = False
            else:
                touch.ud['sv.handled']['x'] = True
            touch.ud['sv.can_defocus'] = False
        if not touch.ud['sv.handled']['y'] and self.do_scroll_y and self.effect_y:
            height = self.height
            if touch.ud.get('in_bar_y', False) and self.vbar[1] != 1.0:
                dy = touch.dy / float(height - height * self.vbar[1])
                self.scroll_y = min(max(self.scroll_y + dy, 0.0), 1.0)
                self._trigger_update_from_scroll()
            elif not_in_bar:
                self.effect_y.update(touch.y)
            if self.scroll_y < 0 or self.scroll_y > 1:
                rv = False
            else:
                touch.ud['sv.handled']['y'] = True
            touch.ud['sv.can_defocus'] = False
        ud['dt'] = touch.time_update - ud['time']
        ud['time'] = touch.time_update
        ud['user_stopped'] = True
    return rv