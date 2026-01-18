import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_banner(self):
    banner = pygame.Rect(0, 0, self.SIZE, BANNER_HEIGHT * CELL_SIZE)
    pygame.draw.rect(self.display, BANNER_COLOR, banner)