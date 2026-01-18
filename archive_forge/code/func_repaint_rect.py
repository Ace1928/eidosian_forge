from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def repaint_rect(self, screen_rect):
    """repaint the given area

        LayeredDirty.repaint_rect(screen_rect): return None

        screen_rect is in screen coordinates.

        """
    if self._clip:
        self.lostsprites.append(screen_rect.clip(self._clip))
    else:
        self.lostsprites.append(Rect(screen_rect))