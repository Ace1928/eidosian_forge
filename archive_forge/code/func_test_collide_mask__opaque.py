import unittest
import pygame
from pygame import sprite
def test_collide_mask__opaque(self):
    self.s1.image.fill((255, 255, 255, 255))
    self.s2.image.fill((255, 255, 255, 255))
    self.s3.image.fill((255, 255, 255, 255))
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_mask), [self.s2])
    self.s1.mask = pygame.mask.from_surface(self.s1.image)
    self.s2.mask = pygame.mask.from_surface(self.s2.image)
    self.s3.mask = pygame.mask.from_surface(self.s3.image)
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_mask), [self.s2])