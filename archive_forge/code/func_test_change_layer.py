import unittest
import pygame
from pygame import sprite
def test_change_layer(self):
    expected_layer = 99
    spr = self.sprite()
    self.LG.add(spr, layer=expected_layer)
    self.assertEqual(self.LG._spritelayers[spr], expected_layer)
    expected_layer = 44
    self.LG.change_layer(spr, expected_layer)
    self.assertEqual(self.LG._spritelayers[spr], expected_layer)
    expected_layer = 77
    spr2 = self.sprite()
    spr2.layer = 55
    self.LG.add(spr2)
    self.LG.change_layer(spr2, expected_layer)
    self.assertEqual(spr2.layer, expected_layer)