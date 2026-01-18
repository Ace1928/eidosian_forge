import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def add_overlay(self, gridpos: int, text1: str, text2: str):
    """Overlays text on the scene."""
    if gridpos not in self._overlays:
        self._overlays[gridpos] = ['', '']
    self._overlays[gridpos][0] += text1 + '\n'
    self._overlays[gridpos][1] += text2 + '\n'