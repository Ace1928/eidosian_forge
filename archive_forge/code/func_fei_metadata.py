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
def fei_metadata(self):
    """Return FEI metadata from SFEG or HELIOS tags as dict."""
    if not self.is_fei:
        return
    tags = self.pages[0].tags
    if 'FEI_SFEG' in tags:
        return tags['FEI_SFEG'].value
    if 'FEI_HELIOS' in tags:
        return tags['FEI_HELIOS'].value