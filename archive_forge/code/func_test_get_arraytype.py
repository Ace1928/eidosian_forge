import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def test_get_arraytype(self):
    array_type = pygame.sndarray.get_arraytype()
    self.assertEqual(array_type, 'numpy', f'unknown array type {array_type}')