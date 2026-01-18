from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def collide_mask(left, right):
    """collision detection between two sprites, using masks.

    pygame.sprite.collide_mask(SpriteLeft, SpriteRight): bool

    Tests for collision between two sprites by testing if their bitmasks
    overlap. If the sprites have a "mask" attribute, that is used as the mask;
    otherwise, a mask is created from the sprite image. Intended to be passed
    as a collided callback function to the *collide functions. Sprites must
    have a "rect" and an optional "mask" attribute.

    New in pygame 1.8.0

    """
    xoffset = right.rect[0] - left.rect[0]
    yoffset = right.rect[1] - left.rect[1]
    try:
        leftmask = left.mask
    except AttributeError:
        leftmask = from_surface(left.image)
    try:
        rightmask = right.mask
    except AttributeError:
        rightmask = from_surface(right.image)
    return leftmask.overlap(rightmask, (xoffset, yoffset))