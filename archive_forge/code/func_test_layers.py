import unittest
import pygame
from pygame import sprite
def test_layers(self):
    sprites = []
    expected_layers = []
    layer_count = 10
    for i in range(layer_count):
        expected_layers.append(i)
        for j in range(5):
            sprites.append(self.sprite())
            sprites[-1]._layer = i
    self.LG.add(sprites)
    layers = self.LG.layers()
    self.assertListEqual(layers, expected_layers)