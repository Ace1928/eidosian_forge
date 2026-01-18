import random
import math
import pygame as pg
def init_star(steps=-1):
    """creates new star values"""
    dir = random.randrange(100000)
    steps_velocity = 1 if steps == -1 else steps * 0.09
    velmult = steps_velocity * (random.random() * 0.6 + 0.4)
    vel = [math.sin(dir) * velmult, math.cos(dir) * velmult]
    if steps is None:
        return [vel, [WINCENTER[0] + vel[0] * steps, WINCENTER[1] + vel[1] * steps]]
    return [vel, WINCENTER[:]]