from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def calculateFitness(self):
    for snake in self.population.saved_snakes:
        fitness = snake.steps ** 3 * 3 ** (snake.score * 3) - 1.5 ** (0.25 * snake.steps)
        snake.fitness = round(fitness, 7)
    self.normalize_fitness_value()