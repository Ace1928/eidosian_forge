from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def normalize_fitness_value(self):
    total_sum = 0
    for snake in self.population.saved_snakes:
        total_sum += snake.fitness
    for snake in self.population.saved_snakes:
        snake.fitness = snake.fitness / total_sum