from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def ate_fruit(self):
    if self.snake.ate_fruit():
        self.snake.add_body_ai()
        self.change_fruit_location()