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
def sem_metadata(self):
    """Return SEM metadata from CZ_SEM tag as dict."""
    if not self.is_sem:
        return
    return self.pages[0].tags['CZ_SEM'].value