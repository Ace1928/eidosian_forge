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
def switch_to(self, screen, **options):
    """Add a new or existing screen to the ScreenManager and switch to it.
        The previous screen will be "switched away" from. `options` are the
        :attr:`transition` options that will be changed before the animation
        happens.

        If no previous screens are available, the screen will be used as the
        main one::

            sm = ScreenManager()
            sm.switch_to(screen1)
            # later
            sm.switch_to(screen2, direction='left')
            # later
            sm.switch_to(screen3, direction='right', duration=1.)

        If any animation is in progress, it will be stopped and replaced by
        this one: you should avoid this because the animation will just look
        weird. Use either :meth:`switch_to` or :attr:`current` but not both.

        The `screen` name will be changed if there is any conflict with the
        current screen.

        .. versionadded: 1.8.0
        """
    assert screen is not None
    if not isinstance(screen, Screen):
        raise ScreenManagerException('ScreenManager accepts only Screen widget.')
    self.transition.stop()
    if screen not in self.screens:
        if self.has_screen(screen.name):
            screen.name = self._generate_screen_name()
    old_transition = self.transition
    specified_transition = options.pop('transition', None)
    if specified_transition:
        self.transition = specified_transition
    for key, value in iteritems(options):
        setattr(self.transition, key, value)
    if screen.manager is not self:
        self.add_widget(screen)
    if self.current_screen is screen:
        return
    old_current = self.current_screen

    def remove_old_screen(transition):
        if old_current in self.children:
            self.remove_widget(old_current)
            self.transition = old_transition
        transition.unbind(on_complete=remove_old_screen)
    self.transition.bind(on_complete=remove_old_screen)
    self.current = screen.name