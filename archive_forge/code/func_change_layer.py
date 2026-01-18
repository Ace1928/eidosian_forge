from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def change_layer(self, sprite, new_layer):
    """change the layer of the sprite

        LayeredUpdates.change_layer(sprite, new_layer): return None

        The sprite must have been added to the renderer already. This is not
        checked.

        """
    LayeredUpdates.change_layer(self, sprite, new_layer)
    if sprite.dirty == 0:
        sprite.dirty = 1