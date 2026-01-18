from abc import ABC, abstractmethod
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
def euclidean_distance(self, nodeA, nodeB):
    distance_1 = nodeA.x - nodeB.x
    distance_2 = nodeA.y - nodeB.y
    return math.sqrt(distance_1 ** 2 + distance_2 ** 2)