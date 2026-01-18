import unittest
import pygame
from pygame import sprite
def test_get_sprites_from_layer(self):
    sprites = {}
    layers = [1, 4, 5, 6, 3, 7, 8, 2, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 6, 5, 4, 3, 2]
    for lay in layers:
        spr = self.sprite()
        spr._layer = lay
        self.LG.add(spr)
        if lay not in sprites:
            sprites[lay] = []
        sprites[lay].append(spr)
    for lay in self.LG.layers():
        for spr in self.LG.get_sprites_from_layer(lay):
            self.assertIn(spr, sprites[lay])
            sprites[lay].remove(spr)
            if len(sprites[lay]) == 0:
                del sprites[lay]
    self.assertEqual(len(sprites.values()), 0)