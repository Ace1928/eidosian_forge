import unittest
import pygame
from pygame import sprite
def test_add__spritelist_passing_layer(self):
    expected_layer = 33
    sprite_count = 10
    sprites = [self.sprite() for _ in range(sprite_count)]
    self.LG.add(sprites, layer=expected_layer)
    self.assertEqual(len(self.LG._spritelist), sprite_count)
    for i in range(sprite_count):
        layer = self.LG.get_layer_of_sprite(sprites[i])
        self.assertEqual(layer, expected_layer)