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
@lazyattr
def is_final(self):
    """Return if page's image data are stored in final form.

        Excludes byte-swapping.

        """
    return self.is_contiguous and self.fillorder == 1 and (self.predictor == 1) and (not self.is_chroma_subsampled)