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
def is_shaped(self):
    """Return description containing array shape if exists, else None."""
    for description in (self.description, self.description1):
        if not description:
            return
        if description[:1] == '{' and '"shape":' in description:
            return description
        if description[:6] == 'shape=':
            return description