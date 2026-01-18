import time
import pygame as pg
from pygame.math import Vector2
from typing import List, Set, Tuple, Optional
from random import randint
import numpy as np
import logging
def is_equal(self, other: 'Node') -> bool:
    return self.position == other.position