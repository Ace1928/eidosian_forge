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
def set_image(current, sliders=sliders, data=data):
    curaxdat[1] = data[tuple(current)].squeeze()
    image.set_data(curaxdat[1])
    for ctrl, index in zip(sliders, current):
        ctrl.eventson = False
        ctrl.set_val(index)
        ctrl.eventson = True
    figure.canvas.draw()