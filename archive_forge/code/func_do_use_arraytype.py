import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def do_use_arraytype(atype):
    pygame.sndarray.use_arraytype(atype)