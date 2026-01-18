import unittest
import pygame
from pygame import sprite
def test_add__sprite_init_layer_attr(self):
    expected_layer = 20
    spr = self.sprite()
    spr._layer = expected_layer
    lrg2 = sprite.LayeredUpdates(spr)
    layer = lrg2._spritelayers[spr]
    self.assertEqual(len(lrg2._spritelist), 1)
    self.assertEqual(layer, expected_layer)