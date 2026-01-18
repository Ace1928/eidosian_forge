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
def andor_tags(self):
    """Return consolidated metadata from Andor tags as dict.

        Remove Andor tags from self.tags.

        """
    if not self.is_andor:
        return
    tags = self.tags
    result = {'Id': tags['AndorId'].value}
    for tag in list(self.tags.values()):
        code = tag.code
        if not 4864 < code < 5031:
            continue
        value = tag.value
        name = tag.name[5:] if len(tag.name) > 5 else tag.name
        result[name] = value
        del tags[tag.name]
    return result