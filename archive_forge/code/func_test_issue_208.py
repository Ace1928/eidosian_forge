import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_issue_208(self):
    """PATCH: pygame.scrap on X11, fix copying into PRIMARY selection

        Copying into theX11 PRIMARY selection (mouse copy/paste) would not
        work due to a confusion between content type and clipboard type.

        """
    from pygame import display, event, freetype
    from pygame.locals import SCRAP_SELECTION, SCRAP_TEXT
    from pygame.locals import KEYDOWN, K_y, QUIT
    success = False
    freetype.init()
    font = freetype.Font(None, 24)
    display.init()
    display.set_caption('Interactive X11 Paste Test')
    screen = display.set_mode((600, 200))
    screen.fill(pygame.Color('white'))
    text = 'Scrap put() succeeded.'
    msg = 'Some text has been placed into the X11 clipboard. Please click the center mouse button in an open text window to retrieve it.\n\nDid you get "{}"? (y/n)'.format(text)
    word_wrap(screen, msg, font, 6)
    display.flip()
    event.pump()
    scrap.init()
    scrap.set_mode(SCRAP_SELECTION)
    scrap.put(SCRAP_TEXT, text.encode('UTF-8'))
    while True:
        e = event.wait()
        if e.type == QUIT:
            break
        if e.type == KEYDOWN:
            success = e.key == K_y
            break
    pygame.display.quit()
    self.assertTrue(success)