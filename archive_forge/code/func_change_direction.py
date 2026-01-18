import apple
from search import SearchAlgorithm
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def change_direction(self, x_direction: int, y_direction: int) -> None:
    if not (abs(self.x_direction - x_direction) == 2 or abs(self.y_direction - y_direction) == 2):
        self.x_direction = x_direction
        self.y_direction = y_direction