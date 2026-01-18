from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def is_cw(coords):
    """Returns True if a polygon ring has clockwise orientation, determined
    by a negatively signed area. 
    """
    area2 = signed_area(coords, fast=True)
    return area2 < 0