from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def groupcollide(groupa, groupb, dokilla, dokillb, collided=None):
    """detect collision between a group and another group

    pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb):
        return dict

    Given two groups, this will find the intersections between all sprites in
    each group. It returns a dictionary of all sprites in the first group that
    collide. The value for each item in the dictionary is a list of the sprites
    in the second group it collides with. The two dokill arguments control if
    the sprites from either group will be automatically removed from all
    groups. Collided is a callback function used to calculate if two sprites
    are colliding. it should take two sprites as values, and return a bool
    value indicating if they are colliding. If collided is not passed, all
    sprites must have a "rect" value, which is a rectangle of the sprite area
    that will be used to calculate the collision.

    """
    crashed = {}
    sprite_collide_func = spritecollide
    if dokilla:
        for group_a_sprite in groupa.sprites():
            collision = sprite_collide_func(group_a_sprite, groupb, dokillb, collided)
            if collision:
                crashed[group_a_sprite] = collision
                group_a_sprite.kill()
    else:
        for group_a_sprite in groupa:
            collision = sprite_collide_func(group_a_sprite, groupb, dokillb, collided)
            if collision:
                crashed[group_a_sprite] = collision
    return crashed