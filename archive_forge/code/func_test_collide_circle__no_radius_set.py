import unittest
import pygame
from pygame import sprite
def test_collide_circle__no_radius_set(self):
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_circle), [self.s2])