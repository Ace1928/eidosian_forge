import os
import pygame as pg
from numpy import zeros, int32, int16
import time
def sound_from_pos(sound, start_pos, samples_per_second=None, inplace=1):
    """returns a sound which begins at the start_pos.
    start_pos - in seconds from the beginning.
    samples_per_second -
    """
    if inplace:
        a1 = pg.sndarray.samples(sound)
    else:
        a1 = pg.sndarray.array(sound)
    if samples_per_second is None:
        samples_per_second = pg.mixer.get_init()[0]
    start_pos_in_samples = int(start_pos * samples_per_second)
    a2 = a1[start_pos_in_samples:]
    sound2 = pg.sndarray.make_sound(a2)
    return sound2