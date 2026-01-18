import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def assign_rel_coord(point, value, invert, coords):
    cx, cy = coords
    if invert:
        value = -1 * value
    if rotation == 0:
        point[cx] += value
    elif rotation == 90:
        point[cy] += value
    elif rotation == 180:
        point[cx] += -value
    elif rotation == 270:
        point[cy] += -value
    point['x'] = min(1.0, max(0.0, point['x']))
    point['y'] = min(1.0, max(0.0, point['y']))