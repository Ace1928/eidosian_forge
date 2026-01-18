import unittest
import pygame
from pygame import sprite
def test_add__spritelist_with_layer_attr(self):
    sprites = []
    sprite_and_layer_count = 10
    for i in range(sprite_and_layer_count):
        sprites.append(self.sprite())
        sprites[-1]._layer = i
    self.LG.add(sprites)
    self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
    for i in range(sprite_and_layer_count):
        layer = self.LG.get_layer_of_sprite(sprites[i])
        self.assertEqual(layer, i)