import unittest
import pygame
from pygame import sprite
def test_collide_rect_ratio__ratio_of_one_like_default(self):
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_rect_ratio(1.0)), [self.s2])