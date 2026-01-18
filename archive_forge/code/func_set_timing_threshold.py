from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def set_timing_threshold(self, time_ms):
    """set the threshold in milliseconds

        set_timing_threshold(time_ms): return None

        Defaults to 1000.0 / 80.0. This means that the screen will be painted
        using the flip method rather than the update method if the update
        method is taking so long to update the screen that the frame rate falls
        below 80 frames per second.

        Raises TypeError if time_ms is not int or float.

        """
    if isinstance(time_ms, (int, float)):
        self._time_threshold = time_ms
    else:
        raise TypeError(f'Expected numeric value, got {time_ms.__class__.__name__} instead')