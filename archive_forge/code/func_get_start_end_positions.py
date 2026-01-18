import time
import pygame as pg
from pygame.math import Vector2
from typing import List, Set, Tuple, Optional
from random import randint
import numpy as np
import logging
def get_start_end_positions(self) -> Tuple[dict, dict]:
    start = {'x': int(self.snake.body[0].x), 'y': int(self.snake.body[0].y)}
    end = {'x': int(self.apple.position[0]), 'y': int(self.apple.position[1])}
    return (start, end)