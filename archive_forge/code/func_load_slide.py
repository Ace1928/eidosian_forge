from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def load_slide(self, slide):
    """Animate to the slide that is passed as the argument.

        .. versionchanged:: 1.8.0
        """
    slides = self.slides
    start, stop = (slides.index(self.current_slide), slides.index(slide))
    if start == stop:
        return
    self._skip_slide = stop
    if stop > start:
        self._prioritize_next = True
        self._insert_visible_slides(_next_slide=slide)
        self.load_next()
    else:
        self._prioritize_next = False
        self._insert_visible_slides(_prev_slide=slide)
        self.load_previous()