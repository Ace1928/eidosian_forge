import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_all_snakes_GA(self):
    if not self.view_path:
        for snake in self.controller.snakes:
            self.draw_snake(snake)
            self.draw_fruit(snake.get_fruit())