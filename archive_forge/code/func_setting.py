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
@udf(HELPER)
def setting(key, value=None):
    if value is None:
        return SETTINGS.get(key)
    else:
        SETTINGS[key] = value
        return value