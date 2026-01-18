import unittest
import pygame
from pygame import sprite
def test_groupcollide__without_collided_callback(self):
    expected_dict = {self.s1: [self.s2]}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
    self.assertDictEqual(expected_dict, crashed)
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
    self.assertDictEqual(expected_dict, crashed)
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
    self.assertDictEqual(expected_dict, crashed)
    self.s3.rect.move_ip(-100, -100)
    expected_dict = {self.s1: [self.s3]}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False)
    self.assertDictEqual(expected_dict, crashed)