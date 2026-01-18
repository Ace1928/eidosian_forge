from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def selectParent(self):
    index = 0
    r = random.random()
    while r > 0:
        r = r - self.population.saved_snakes[index].fitness
        index += 1
    index -= 1
    return self.population.saved_snakes[index]