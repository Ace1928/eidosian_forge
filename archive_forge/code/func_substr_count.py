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
@udf(STRING)
def substr_count(haystack, needle):
    if not haystack or not needle:
        return 0
    return haystack.count(needle)