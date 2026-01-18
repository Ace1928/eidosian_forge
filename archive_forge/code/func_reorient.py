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
def reorient(image, orientation):
    """Return reoriented view of image array.

    Parameters
    ----------
    image : numpy.ndarray
        Non-squeezed output of asarray() functions.
        Axes -3 and -2 must be image length and width respectively.
    orientation : int or str
        One of TIFF.ORIENTATION names or values.

    """
    ORIENTATION = TIFF.ORIENTATION
    orientation = enumarg(ORIENTATION, orientation)
    if orientation == ORIENTATION.TOPLEFT:
        return image
    elif orientation == ORIENTATION.TOPRIGHT:
        return image[..., ::-1, :]
    elif orientation == ORIENTATION.BOTLEFT:
        return image[..., ::-1, :, :]
    elif orientation == ORIENTATION.BOTRIGHT:
        return image[..., ::-1, ::-1, :]
    elif orientation == ORIENTATION.LEFTTOP:
        return numpy.swapaxes(image, -3, -2)
    elif orientation == ORIENTATION.RIGHTTOP:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, :]
    elif orientation == ORIENTATION.RIGHTBOT:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, :, :]
    elif orientation == ORIENTATION.LEFTBOT:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, ::-1, :]