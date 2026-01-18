import os
import time
import unittest
import pygame
import pygame.key
def test_name_and_key_code(self):
    for const_name in dir(pygame):
        if not const_name.startswith('K_') or const_name in SKIPPED_KEYS:
            continue
        try:
            expected_str_name = KEY_NAME_COMPAT[const_name]
        except KeyError:
            self.fail('If you are seeing this error in a test run, you probably added a new pygame key constant, but forgot to update key_test unitests')
        const_val = getattr(pygame, const_name)
        self.assertEqual(pygame.key.name(const_val), expected_str_name)
        self.assertEqual(pygame.key.name(key=const_val), expected_str_name)
        self.assertEqual(pygame.key.key_code(expected_str_name), const_val)
        self.assertEqual(pygame.key.key_code(name=expected_str_name), const_val)
        alt_name = pygame.key.name(const_val, use_compat=False)
        self.assertIsInstance(alt_name, str)
        self.assertEqual(pygame.key.key_code(alt_name), const_val)
    self.assertRaises(TypeError, pygame.key.name, 'fizzbuzz')
    self.assertRaises(TypeError, pygame.key.key_code, pygame.K_a)
    self.assertRaises(ValueError, pygame.key.key_code, 'fizzbuzz')