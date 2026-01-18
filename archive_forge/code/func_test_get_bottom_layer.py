import unittest
import pygame
from pygame import sprite
def test_get_bottom_layer(self):
    layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
    for i in layers:
        self.LG.add(self.sprite(), layer=i)
    bottom_layer = self.LG.get_bottom_layer()
    self.assertEqual(bottom_layer, self.LG.get_bottom_layer())
    self.assertEqual(bottom_layer, min(layers))
    self.assertEqual(bottom_layer, min(self.LG._spritelayers.values()))
    self.assertEqual(bottom_layer, self.LG._spritelayers[self.LG._spritelist[0]])