import unittest
import pygame
from pygame import sprite
def test_move_to_back(self):
    layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
    for i in layers:
        self.LG.add(self.sprite(), layer=i)
    spr = self.sprite()
    self.LG.add(spr, layer=55)
    self.assertNotEqual(spr, self.LG._spritelist[0])
    self.LG.move_to_back(spr)
    self.assertEqual(spr, self.LG._spritelist[0])