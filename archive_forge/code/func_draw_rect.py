import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_rect(self, element, color, border=False):
    x = int(element.x * CELL_SIZE)
    y = int(element.y * CELL_SIZE)
    body_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(self.display, color, body_rect)
    if border:
        pygame.draw.rect(self.display, WINDOW_COLOR, body_rect, 3)