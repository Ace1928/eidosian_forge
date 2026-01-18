from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNework
import pickle
def ate_body(self):
    for body in self.body[1:]:
        if self.body[0] == body:
            return True
    return False