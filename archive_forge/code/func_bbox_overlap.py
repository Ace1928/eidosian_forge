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
def bbox_overlap(bbox1, bbox2):
    """Tests whether two bounding boxes overlap, returning a boolean
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    overlap = xmin1 <= xmax2 and xmax1 >= xmin2 and (ymin1 <= ymax2) and (ymax1 >= ymin2)
    return overlap