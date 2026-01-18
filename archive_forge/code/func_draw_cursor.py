import pygame
from Constants import *
from GA import *
import sys
def draw_cursor(self):
    self.game.draw_text('*', size=20, x=self.cursor_rect.x, y=self.cursor_rect.y, color=MENU_COLOR)