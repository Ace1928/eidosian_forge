import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def add_marker(self, **marker_params):
    self._markers.append(marker_params)