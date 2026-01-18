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
@udf(DATE)
def strip_tz(date_str):
    date_str = date_str.replace('T', ' ')
    tz_idx1 = date_str.find('+')
    if tz_idx1 != -1:
        return date_str[:tz_idx1]
    tz_idx2 = date_str.find('-')
    if tz_idx2 > 13:
        return date_str[:tz_idx2]
    return date_str