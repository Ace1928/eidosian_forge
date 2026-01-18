import random
import math
import pygame as pg
def initialize_stars():
    """creates a new starfield"""
    random.seed()
    stars = [init_star(steps=random.randint(0, WINCENTER[0])) for _ in range(NUMSTARS)]
    move_stars(stars)
    return stars