from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def get_best_snake(self):
    best_snake = self.population.saved_snakes[0]
    for snake in self.population.saved_snakes:
        if snake.fitness > best_snake.fitness:
            best_snake = snake
    if best_snake.score > self.best_score:
        self.best_score = best_snake.score
        self.best_gen = self.generation
        self.best_snake = best_snake
    return best_snake