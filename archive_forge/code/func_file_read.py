import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@udf(FILE)
def file_read(filename):
    try:
        with open(filename) as fh:
            return fh.read()
    except:
        pass