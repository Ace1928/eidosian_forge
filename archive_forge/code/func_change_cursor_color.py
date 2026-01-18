import pygame
from Constants import *
from GA import *
import sys
def change_cursor_color(self):
    self.clear_cursor_color()
    if self.state == 'BFS':
        self.cursorBFS = MENU_COLOR
    elif self.state == 'DFS':
        self.cursorDFS = MENU_COLOR
    elif self.state == 'ASTAR':
        self.cursorASTAR = MENU_COLOR
    elif self.state == 'GA':
        self.cursorGA = MENU_COLOR