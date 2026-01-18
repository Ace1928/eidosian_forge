import threading
import time
import numpy as np
import pygame as pg
from moviepy.decorators import convert_masks_to_RGB, requires_duration
from moviepy.tools import cvsecs
def imdisplay(imarray, screen=None):
    """Splashes the given image array on the given pygame screen """
    a = pg.surfarray.make_surface(imarray.swapaxes(0, 1))
    if screen is None:
        screen = pg.display.set_mode(imarray.shape[:2][::-1])
    screen.blit(a, (0, 0))
    pg.display.flip()