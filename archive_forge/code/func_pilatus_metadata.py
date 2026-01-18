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
def pilatus_metadata(self):
    """Return Pilatus metadata from image description as dict."""
    if not self.is_pilatus:
        return
    return pilatus_description_metadata(self.pages[0].description)