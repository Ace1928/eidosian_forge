import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_aaline__valid_end_pos_formats(self):
    """Ensures draw aaline accepts different end_pos formats."""
    expected_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((4, 4))
    kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None}
    x, y = (2, 2)
    positions = ((x, y), (x + 0.02, y), (x, y + 0.02), (x + 0.02, y + 0.02))
    for end_pos in positions:
        for seq_type in (tuple, list, Vector2):
            surface.fill(surface_color)
            kwargs['end_pos'] = seq_type(end_pos)
            bounds_rect = self.draw_aaline(**kwargs)
            color = surface.get_at((x, y))
            for i, sub_color in enumerate(expected_color):
                self.assertGreaterEqual(color[i] + 15, sub_color, end_pos)
            self.assertIsInstance(bounds_rect, pygame.Rect, end_pos)