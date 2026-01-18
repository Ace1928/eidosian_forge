from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def change_fruit_location(self):
    while True:
        self.snake.create_fruit()
        position = self.snake.get_fruit()
        inside_body = False
        for body in self.snake.body:
            if position == body:
                inside_body = True
        if inside_body == False:
            break