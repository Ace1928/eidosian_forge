import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test__get_count_interactive(self):
    prompt('Please connect at least one controller before the test for controller.get_count() starts')
    controller.quit()
    controller.init()
    joystick_num = controller.get_count()
    ans = question('get_count() thinks there are {} joysticks connected. Is that correct?'.format(joystick_num))
    self.assertTrue(ans)