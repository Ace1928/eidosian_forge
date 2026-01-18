import pygame
from Constants import *
from GA import *
import sys
def blit_menu(self):
    self.game.window.blit(self.game.display, (0, 0))
    pygame.display.update()
    self.game.reset_keys()