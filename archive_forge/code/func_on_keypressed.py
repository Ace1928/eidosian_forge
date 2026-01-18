from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def on_keypressed(event, data=data, current=current):
    key = event.key
    axis = curaxdat[0]
    if str(key) in '0123456789':
        on_changed(key, axis)
    elif key == 'right':
        on_changed(current[axis] + 1, axis)
    elif key == 'left':
        on_changed(current[axis] - 1, axis)
    elif key == 'up':
        curaxdat[0] = 0 if axis == len(data.shape) - 1 else axis + 1
    elif key == 'down':
        curaxdat[0] = len(data.shape) - 1 if axis == 0 else axis - 1
    elif key == 'end':
        on_changed(data.shape[axis] - 1, axis)
    elif key == 'home':
        on_changed(0, axis)