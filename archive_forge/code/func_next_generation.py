from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def next_generation(self):
    if self.generation == GA.generation:
        return False
    self.calculateFitness()
    self.get_best_snake()
    self.naturalSelection()
    self.population.saved_snakes = []
    return True