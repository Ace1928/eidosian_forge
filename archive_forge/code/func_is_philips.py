from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@property
def is_philips(self) -> bool:
    """Page contains Philips DP metadata."""
    return self.software[:10] == 'Philips DP' and self.description[-16:].strip().endswith('</DataObject>')