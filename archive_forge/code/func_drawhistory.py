from typing import List
import pygame as pg
import pygame._sdl2.controller
def drawhistory(win, history):
    img = font.render('Event History Area', 1, (155, 155, 155), (0, 0, 0))
    win.blit(img, (2, 132))
    ypos = 450
    h = list(history)
    h.reverse()
    for line in h:
        r = win.blit(line, (10, ypos))
        win.fill(0, (r.right, r.top, 620, r.height))
        ypos -= font.get_height()