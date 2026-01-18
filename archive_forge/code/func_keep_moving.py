from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def keep_moving(self):
    x = self.snake.body[0].x
    y = self.snake.body[0].y
    if self.snake.body[1].x == x:
        if self.snake.body[1].y < y:
            y = y + 1
        else:
            y = y - 1
    elif self.snake.body[1].y == y:
        if self.snake.body[1].x < x:
            x = x + 1
        else:
            x = x - 1
    return (x, y)