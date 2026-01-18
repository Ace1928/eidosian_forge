import unittest
import pygame
from pygame import sprite
def test_get_sprites_at(self):
    sprites = []
    expected_sprites = []
    for i in range(3):
        spr = self.sprite()
        spr.rect = pygame.Rect(i * 50, i * 50, 100, 100)
        sprites.append(spr)
        if i < 2:
            expected_sprites.append(spr)
    self.LG.add(sprites)
    result = self.LG.get_sprites_at((50, 50))
    self.assertEqual(result, expected_sprites)