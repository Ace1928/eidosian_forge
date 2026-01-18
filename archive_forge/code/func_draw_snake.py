import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_snake(self, snake):
    self.draw_snake_head(snake)
    for body in snake.body[1:]:
        self.draw_snake_body(body)