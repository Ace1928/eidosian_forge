import unittest
import pygame
from pygame import sprite
def test_collide_circle_ratio__no_radius_and_ratio_of_twenty(self):
    collided_func = sprite.collide_circle_ratio(20.0)
    expected_sprites = sorted(self.ag2.sprites(), key=id)
    collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
    self.assertListEqual(expected_sprites, collided_sprites)