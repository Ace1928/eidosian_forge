import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
def test_gl_get_attribute(self):
    screen = display.set_mode((0, 0), pygame.OPENGL)
    original_values = []
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_RED_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_GREEN_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_BLUE_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_ALPHA_SIZE))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_STEREO))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCELERATED_VISUAL))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_FLAGS))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_PROFILE_MASK))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
    original_values.append(pygame.display.gl_get_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
    pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
    pygame.display.gl_set_attribute(pygame.GL_ACCUM_RED_SIZE, 16)
    pygame.display.gl_set_attribute(pygame.GL_ACCUM_GREEN_SIZE, 16)
    pygame.display.gl_set_attribute(pygame.GL_ACCUM_BLUE_SIZE, 16)
    pygame.display.gl_set_attribute(pygame.GL_ACCUM_ALPHA_SIZE, 16)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 1)
    pygame.display.gl_set_attribute(pygame.GL_STEREO, 0)
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, 0)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 0)
    pygame.display.gl_set_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT, 0)
    pygame.display.gl_set_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE, 0)
    set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    get_values = []
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_RED_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_GREEN_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_BLUE_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_ALPHA_SIZE))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_STEREO))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCELERATED_VISUAL))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_FLAGS))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_PROFILE_MASK))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
    get_values.append(pygame.display.gl_get_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
    for i in range(len(original_values)):
        self.assertTrue(get_values[i] == original_values[i] or get_values[i] == set_values[i])
    with self.assertRaises(TypeError):
        pygame.display.gl_get_attribute('DUMMY')