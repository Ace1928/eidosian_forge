import os
import sys
import pygame as pg
def scaleit(fin, fout, w, h):
    i = pg.image.load(fin)
    if hasattr(pg.transform, 'smoothscale'):
        scaled_image = pg.transform.smoothscale(i, (w, h))
    else:
        scaled_image = pg.transform.scale(i, (w, h))
    pg.image.save(scaled_image, fout)