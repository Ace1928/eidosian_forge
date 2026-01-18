from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def naturalSelection(self):
    new_snakes = []
    for i in range(Population.population):
        parentA = self.selectParent()
        parentB = self.selectParent()
        child = Snake(Population.hidden_node)
        child.network.crossover(parentA.network, parentB.network)
        child.network.mutate(GA.mutation_rate)
        new_snakes.append(child)
    self.population.snakes = new_snakes.copy()
    self.generation += 1