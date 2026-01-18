import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def assign_coord(point, value, invert, coords):
    cx, cy = coords
    if invert:
        value = 1.0 - value
    if rotation == 0:
        point[cx] = value
    elif rotation == 90:
        point[cy] = value
    elif rotation == 180:
        point[cx] = 1.0 - value
    elif rotation == 270:
        point[cy] = 1.0 - value