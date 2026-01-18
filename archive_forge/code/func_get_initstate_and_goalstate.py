from abc import ABC, abstractmethod
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
def get_initstate_and_goalstate(self, snake):
    return (Node(snake.get_x(), snake.get_y()), Node(snake.get_fruit().x, snake.get_fruit().y))