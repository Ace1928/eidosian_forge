from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def ai_play(self, algorithm):
    self.set_algorithm(algorithm)
    if self.algo == None:
        return
    if isinstance(self.algo, GA):
        self.update_GA_ai()
    else:
        pos = self.algo.run_algorithm(self.snake)
        self.update_path_finding_algo(pos)