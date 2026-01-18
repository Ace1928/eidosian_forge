import pygame as pg
import time
from pygame._sdl2 import (
from pygame._sdl2.mixer import set_post_mix
def postmix_callback(postmix, audiomemoryview):
    """This is called in the sound thread.

    At the end of mixing we get this data.
    """
    print(type(audiomemoryview), len(audiomemoryview))
    print(postmix)