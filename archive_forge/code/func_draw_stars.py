import random
import math
import pygame as pg
def draw_stars(surface, stars, color):
    """used to draw (and clear) the stars"""
    for _, pos in stars:
        pos = (int(pos[0]), int(pos[1]))
        surface.set_at(pos, color)