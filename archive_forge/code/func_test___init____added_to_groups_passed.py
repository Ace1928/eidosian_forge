import unittest
import pygame
from pygame import sprite
def test___init____added_to_groups_passed(self):
    expected_groups = sorted(self.groups, key=id)
    sprite = self.Sprite(self.groups)
    groups = sorted(sprite.groups(), key=id)
    self.assertListEqual(groups, expected_groups)