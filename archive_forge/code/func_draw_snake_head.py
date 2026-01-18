import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_snake_head(self, snake):
    head = snake.body[0]
    self.draw_rect(head, color=SNAKE_HEAD_COLOR)