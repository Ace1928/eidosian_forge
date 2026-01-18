import os
import pygame as pg
def load_sound(name):

    class NoneSound:

        def play(self):
            pass
    if not pg.mixer or not pg.mixer.get_init():
        return NoneSound()
    fullname = os.path.join(data_dir, name)
    sound = pg.mixer.Sound(fullname)
    return sound