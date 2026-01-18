from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_top_sprite(self):
    """return the topmost sprite

        LayeredUpdates.get_top_sprite(): return Sprite

        """
    return self._spritelist[-1]