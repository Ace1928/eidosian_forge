from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_layer_of_sprite(self, sprite):
    """return the layer that sprite is currently in

        If the sprite is not found, then it will return the default layer.

        """
    return self._spritelayers.get(sprite, self._default_layer)