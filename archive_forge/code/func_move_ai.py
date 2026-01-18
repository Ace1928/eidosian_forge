from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNework
import pickle
def move_ai(self, x, y):
    self.life_time += 1
    self.steps += 1
    for i in range(len(self.body) - 1, 0, -1):
        self.body[i].x = self.body[i - 1].x
        self.body[i].y = self.body[i - 1].y
    self.body[0].x = x
    self.body[0].y = y