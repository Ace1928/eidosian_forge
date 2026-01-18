from abc import ABC, abstractmethod
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
def outside_boundary(self, node):
    if not 0 <= node.x < NO_OF_CELLS:
        return True
    elif not BANNER_HEIGHT <= node.y < NO_OF_CELLS:
        return True
    return False