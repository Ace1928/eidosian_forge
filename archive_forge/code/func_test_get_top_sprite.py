import unittest
import pygame
from pygame import sprite
def test_get_top_sprite(self):
    layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
    for i in layers:
        self.LG.add(self.sprite(), layer=i)
    expected_layer = self.LG.get_top_layer()
    layer = self.LG.get_layer_of_sprite(self.LG.get_top_sprite())
    self.assertEqual(layer, expected_layer)