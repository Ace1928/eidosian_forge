import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
def test_gl_set_attribute_kwargs(self):
    screen = display.set_mode((0, 0), pygame.OPENGL)
    set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0]
    pygame.display.gl_set_attribute(flag=pygame.GL_ALPHA_SIZE, value=set_values[0])
    pygame.display.gl_set_attribute(flag=pygame.GL_DEPTH_SIZE, value=set_values[1])
    pygame.display.gl_set_attribute(flag=pygame.GL_STENCIL_SIZE, value=set_values[2])
    pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_RED_SIZE, value=set_values[3])
    pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE, value=set_values[4])
    pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE, value=set_values[5])
    pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE, value=set_values[6])
    pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS, value=set_values[7])
    pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLESAMPLES, value=set_values[8])
    pygame.display.gl_set_attribute(flag=pygame.GL_STEREO, value=set_values[9])
    get_values = []
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ALPHA_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_DEPTH_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STENCIL_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_RED_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLESAMPLES))
    get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STEREO))
    for i in range(len(set_values)):
        self.assertTrue(get_values[i] == set_values[i])
    with self.assertRaises(TypeError):
        pygame.display.gl_get_attribute('DUMMY')