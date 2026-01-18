from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def collide_rect(left, right):
    """collision detection between two sprites, using rects.

    pygame.sprite.collide_rect(left, right): return bool

    Tests for collision between two sprites. Uses the pygame.Rect colliderect
    function to calculate the collision. It is intended to be passed as a
    collided callback function to the *collide functions. Sprites must have
    "rect" attributes.

    New in pygame 1.8.0

    """
    return left.rect.colliderect(right.rect)