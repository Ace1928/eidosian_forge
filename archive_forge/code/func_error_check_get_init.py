import unittest
from pygame.tests.test_utils import question, prompt
import pygame
import pygame._sdl2.controller
def error_check_get_init():
    try:
        pygame.joystick.get_count()
    except pygame.error:
        return False
    return True