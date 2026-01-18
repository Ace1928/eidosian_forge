import unittest
import pygame
from pygame.locals import *
def make_blit_list(self, num_surfs):
    blit_list = []
    for i in range(num_surfs):
        dest = (i * 10, 0)
        surf = pygame.Surface((10, 10), SRCALPHA, 32)
        color = (i * 1, i * 1, i * 1)
        surf.fill(color)
        blit_list.append((surf, dest))
    return blit_list