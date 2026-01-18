from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def check_directions(self, snake, directions, inputs):
    if self.outside_boundary(directions) or self.inside_body(snake, directions):
        inputs.append(1)
    else:
        inputs.append(0)