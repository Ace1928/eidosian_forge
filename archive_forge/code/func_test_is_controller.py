import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_is_controller(self):
    for i in range(controller.get_count()):
        if controller.is_controller(i):
            c = controller.Controller(i)
            self.assertIsInstance(c, controller.Controller)
            c.quit()
        else:
            with self.assertRaises(pygame._sdl2.sdl2.error):
                c = controller.Controller(i)
    with self.assertRaises(TypeError):
        controller.is_controller('Test')