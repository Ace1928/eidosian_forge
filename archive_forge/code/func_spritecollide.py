from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def spritecollide(sprite, group, dokill, collided=None):
    """find Sprites in a Group that intersect another Sprite

    pygame.sprite.spritecollide(sprite, group, dokill, collided=None):
        return Sprite_list

    Return a list containing all Sprites in a Group that intersect with another
    Sprite. Intersection is determined by comparing the Sprite.rect attribute
    of each Sprite.

    The dokill argument is a bool. If set to True, all Sprites that collide
    will be removed from the Group.

    The collided argument is a callback function used to calculate if two
    sprites are colliding. it should take two sprites as values, and return a
    bool value indicating if they are colliding. If collided is not passed, all
    sprites must have a "rect" value, which is a rectangle of the sprite area,
    which will be used to calculate the collision.

    """
    default_sprite_collide_func = sprite.rect.colliderect
    if dokill:
        crashed = []
        append = crashed.append
        for group_sprite in group.sprites():
            if collided is not None:
                if collided(sprite, group_sprite):
                    group_sprite.kill()
                    append(group_sprite)
            elif default_sprite_collide_func(group_sprite.rect):
                group_sprite.kill()
                append(group_sprite)
        return crashed
    if collided is not None:
        return [group_sprite for group_sprite in group if collided(sprite, group_sprite)]
    return [group_sprite for group_sprite in group if default_sprite_collide_func(group_sprite.rect)]