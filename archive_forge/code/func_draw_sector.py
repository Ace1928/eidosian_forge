import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def draw_sector(self, center: point_like, point: point_like, beta: float, fullSector: bool=True):
    """Draw a circle sector."""
    center = Point(center)
    point = Point(point)
    l3 = '%g %g m\n'
    l4 = '%g %g %g %g %g %g c\n'
    l5 = '%g %g l\n'
    betar = math.radians(-beta)
    w360 = math.radians(math.copysign(360, betar)) * -1
    w90 = math.radians(math.copysign(90, betar))
    w45 = w90 / 2
    while abs(betar) > 2 * math.pi:
        betar += w360
    if not self.last_point == point:
        self.draw_cont += l3 % JM_TUPLE(point * self.ipctm)
        self.last_point = point
    Q = Point(0, 0)
    C = center
    P = point
    S = P - C
    rad = abs(S)
    if not rad > EPSILON:
        raise ValueError('radius must be positive')
    alfa = self.horizontal_angle(center, point)
    while abs(betar) > abs(w90):
        q1 = C.x + math.cos(alfa + w90) * rad
        q2 = C.y + math.sin(alfa + w90) * rad
        Q = Point(q1, q2)
        r1 = C.x + math.cos(alfa + w45) * rad / math.cos(w45)
        r2 = C.y + math.sin(alfa + w45) * rad / math.cos(w45)
        R = Point(r1, r2)
        kappah = (1 - math.cos(w45)) * 4 / 3 / abs(R - Q)
        kappa = kappah * abs(P - Q)
        cp1 = P + (R - P) * kappa
        cp2 = Q + (R - Q) * kappa
        self.draw_cont += l4 % JM_TUPLE(list(cp1 * self.ipctm) + list(cp2 * self.ipctm) + list(Q * self.ipctm))
        betar -= w90
        alfa += w90
        P = Q
    if abs(betar) > 0.001:
        beta2 = betar / 2
        q1 = C.x + math.cos(alfa + betar) * rad
        q2 = C.y + math.sin(alfa + betar) * rad
        Q = Point(q1, q2)
        r1 = C.x + math.cos(alfa + beta2) * rad / math.cos(beta2)
        r2 = C.y + math.sin(alfa + beta2) * rad / math.cos(beta2)
        R = Point(r1, r2)
        kappah = (1 - math.cos(beta2)) * 4 / 3 / abs(R - Q)
        kappa = kappah * abs(P - Q) / (1 - math.cos(betar))
        cp1 = P + (R - P) * kappa
        cp2 = Q + (R - Q) * kappa
        self.draw_cont += l4 % JM_TUPLE(list(cp1 * self.ipctm) + list(cp2 * self.ipctm) + list(Q * self.ipctm))
    if fullSector:
        self.draw_cont += l3 % JM_TUPLE(point * self.ipctm)
        self.draw_cont += l5 % JM_TUPLE(center * self.ipctm)
        self.draw_cont += l5 % JM_TUPLE(Q * self.ipctm)
    self.last_point = Q
    return self.last_point