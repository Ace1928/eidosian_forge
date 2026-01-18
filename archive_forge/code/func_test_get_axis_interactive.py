import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_get_axis_interactive(self):
    c = self._get_first_controller()
    if not c:
        self.skipTest('No controller connected')
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((400, 400))
    font = pygame.font.Font(None, 20)
    running = True
    label1 = font.render('Press down the right trigger. The value on-screen should', True, (0, 0, 0))
    label2 = font.render('indicate how far the trigger is pressed down. This value should', True, (0, 0, 0))
    label3 = font.render('be in the range of 0-32767. Press "y" or "n" to confirm.', True, (0, 0, 0))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    running = False
                if event.key == pygame.K_n:
                    running = False
                    pygame.display.quit()
                    pygame.font.quit()
                    self.fail()
        right_trigger = c.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
        screen.fill((255, 255, 255))
        screen.blit(label1, (0, 0))
        screen.blit(label2, (0, 20))
        screen.blit(label3, (0, 40))
        screen.blit(font.render(str(right_trigger), True, (0, 0, 0)), (0, 60))
        pygame.display.update()
    pygame.display.quit()
    pygame.font.quit()