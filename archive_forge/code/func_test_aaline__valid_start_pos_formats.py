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
def test_aaline__valid_start_pos_formats(self):
    """Ensures draw aaline accepts different start_pos formats."""
    expected_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((4, 4))
    kwargs = {'surface': surface, 'color': expected_color, 'start_pos': None, 'end_pos': (2, 2)}
    x, y = (2, 1)
    positions = ((x, y), (x + 0.01, y), (x, y + 0.01), (x + 0.01, y + 0.01))
    for start_pos in positions:
        for seq_type in (tuple, list, Vector2):
            surface.fill(surface_color)
            kwargs['start_pos'] = seq_type(start_pos)
            bounds_rect = self.draw_aaline(**kwargs)
            color = surface.get_at((x, y))
            for i, sub_color in enumerate(expected_color):
                self.assertGreaterEqual(color[i] + 6, sub_color, start_pos)
            self.assertIsInstance(bounds_rect, pygame.Rect, start_pos)