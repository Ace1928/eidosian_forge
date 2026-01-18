from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def on_bring_to_front(self, touch):
    """
        Called when a touch event causes the scatter to be brought to the
        front of the parent (only if :attr:`auto_bring_to_front` is True)

        :Parameters:
            `touch`:
                The touch object which brought the scatter to front.

        .. versionadded:: 1.9.0
        """
    pass