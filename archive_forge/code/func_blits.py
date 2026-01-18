import unittest
import pygame
from pygame.locals import *
def blits(blit_list):
    for surface, dest in blit_list:
        dst.blit(surface, dest)