import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_from_joystick(self):
    for i in range(controller.get_count()):
        if controller.is_controller(i):
            joy = pygame.joystick.Joystick(i)
            break
    else:
        self.skipTest('No controller connected')
    c = controller.Controller.from_joystick(joy)
    self.assertIsInstance(c, controller.Controller)