import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') in ['dummy', 'android'], 'iconify is only supported on some video drivers/platforms')
def test_iconify(self):
    pygame.display.set_mode((640, 480))
    self.assertEqual(pygame.display.get_active(), True)
    success = pygame.display.iconify()
    if success:
        active_event = window_minimized_event = False
        for _ in range(50):
            time.sleep(0.01)
            for event in pygame.event.get():
                if event.type == pygame.ACTIVEEVENT:
                    if not event.gain and event.state == pygame.APPACTIVE:
                        active_event = True
                if event.type == pygame.WINDOWMINIMIZED:
                    window_minimized_event = True
        self.assertTrue(window_minimized_event)
        self.assertTrue(active_event)
        self.assertFalse(pygame.display.get_active())
    else:
        self.fail('Iconify not supported on this platform, please skip')