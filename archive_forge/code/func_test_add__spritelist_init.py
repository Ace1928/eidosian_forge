import unittest
import pygame
from pygame import sprite
def test_add__spritelist_init(self):
    sprite_count = 10
    sprites = [self.sprite() for _ in range(sprite_count)]
    lrg2 = sprite.LayeredUpdates(sprites)
    expected_layer = lrg2._default_layer
    self.assertEqual(len(lrg2._spritelist), sprite_count)
    for i in range(sprite_count):
        layer = lrg2.get_layer_of_sprite(sprites[i])
        self.assertEqual(layer, expected_layer)