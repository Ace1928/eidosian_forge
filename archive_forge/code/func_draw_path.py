import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_path(self):
    if self.controller.algo != None and self.view_path:
        for path in self.controller.algo.path:
            x = int(path.x * CELL_SIZE)
            y = int(path.y * CELL_SIZE)
            path_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            shape_surf = pygame.Surface(path_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, PATHCOLOR, shape_surf.get_rect())
            pygame.draw.rect(self.display, BANNER_COLOR, path_rect, 1)
            self.display.blit(shape_surf, path_rect)