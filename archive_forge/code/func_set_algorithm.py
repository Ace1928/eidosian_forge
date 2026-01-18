from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def set_algorithm(self, algo_type):
    if self.algo != None:
        return
    if algo_type == 'BFS':
        self.algo = BFS(self.grid)
        self.snake = Snake()
    elif algo_type == 'DFS':
        self.algo = DFS(self.grid)
        self.snake = Snake()
    elif algo_type == 'ASTAR':
        self.algo = A_STAR(self.grid)
        self.snake = Snake()
    elif algo_type == 'GA':
        self.algo = GA(self.grid)
        if not self.model_loaded:
            self.algo.population._initialpopulation_()
            self.snakes = self.algo.population.snakes