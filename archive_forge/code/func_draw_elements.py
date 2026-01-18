import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_elements(self):
    self.draw_banner()
    self.draw_game_stats()
    if self.curr_menu.state != 'GA' or self.controller.model_loaded:
        fruit = self.controller.get_fruit_pos()
        snake = self.controller.snake
        self.draw_fruit(fruit)
        self.draw_snake(snake)
        self.draw_score()
        if not self.controller.model_loaded:
            self.draw_path()
    else:
        self.draw_all_snakes_GA()