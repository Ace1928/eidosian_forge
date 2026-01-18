import unittest
import pygame
from pygame import sprite
def test_groupcollide__with_collided_callback(self):
    collided_callback_true = lambda spr_a, spr_b: True
    collided_callback_false = lambda spr_a, spr_b: False
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_false)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {self.s1: sorted(self.ag2.sprites(), key=id)}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_true)
    for value in crashed.values():
        value.sort(key=id)
    self.assertDictEqual(expected_dict, crashed)
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, False, collided_callback_true)
    for value in crashed.values():
        value.sort(key=id)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_false)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {self.s1: sorted(self.ag2.sprites(), key=id)}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_true)
    for value in crashed.values():
        value.sort(key=id)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, False, True, collided_callback_true)
    self.assertDictEqual(expected_dict, crashed)
    self.ag.add(self.s2)
    self.ag2.add(self.s3)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_false)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {self.s1: [self.s3], self.s2: [self.s3]}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_true)
    self.assertDictEqual(expected_dict, crashed)
    expected_dict = {}
    crashed = pygame.sprite.groupcollide(self.ag, self.ag2, True, False, collided_callback_true)
    self.assertDictEqual(expected_dict, crashed)