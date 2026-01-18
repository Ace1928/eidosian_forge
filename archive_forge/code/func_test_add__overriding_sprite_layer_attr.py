import unittest
import pygame
from pygame import sprite
def test_add__overriding_sprite_layer_attr(self):
    expected_layer = 200
    spr = self.sprite()
    spr._layer = 100
    self.LG.add(spr, layer=expected_layer)
    layer = self.LG.get_layer_of_sprite(spr)
    self.assertEqual(len(self.LG._spritelist), 1)
    self.assertEqual(layer, expected_layer)