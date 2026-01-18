from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def set_clip(self, screen_rect=None):
    """clip the area where to draw; pass None (default) to reset the clip

        LayeredDirty.set_clip(screen_rect=None): return None

        """
    if screen_rect is None:
        self._clip = pygame.display.get_surface().get_rect()
    else:
        self._clip = screen_rect
    self._use_update = False