import unittest
import pygame
from pygame import sprite
def test_add__adding_sprite_on_init(self):
    spr = self.sprite()
    lrg2 = sprite.LayeredUpdates(spr)
    expected_layer = lrg2._default_layer
    layer = lrg2._spritelayers[spr]
    self.assertEqual(len(lrg2._spritelist), 1)
    self.assertEqual(layer, expected_layer)