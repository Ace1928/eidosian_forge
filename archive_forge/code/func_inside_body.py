from abc import ABC, abstractmethod
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
def inside_body(self, snake, node):
    for body in snake.body:
        if body.x == node.x and body.y == node.y:
            return True
    return False