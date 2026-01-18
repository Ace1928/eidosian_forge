from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_sprites_from_layer(self, layer):
    """return all sprites from a layer ordered as they where added

        LayeredUpdates.get_sprites_from_layer(layer): return sprites

        Returns all sprites from a layer. The sprites are ordered in the
        sequence that they where added. (The sprites are not removed from the
        layer.

        """
    sprites = []
    sprites_append = sprites.append
    sprite_layers = self._spritelayers
    for spr in self._spritelist:
        if sprite_layers[spr] == layer:
            sprites_append(spr)
        elif sprite_layers[spr] > layer:
            break
    return sprites