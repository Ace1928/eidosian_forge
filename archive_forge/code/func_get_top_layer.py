from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_top_layer(self):
    """return the top layer

        LayeredUpdates.get_top_layer(): return layer

        """
    return self._spritelayers[self._spritelist[-1]]