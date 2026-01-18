import os
import zlib
import time  # noqa
import logging
import numpy as np
def make_rect_record(self, xmin, xmax, ymin, ymax):
    """Simply uses makeCompactArray to produce
        a RECT Record."""
    return twits2bits([xmin, xmax, ymin, ymax])