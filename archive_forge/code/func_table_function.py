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
def table_function(*groups):

    def decorator(klass):
        for group in groups:
            TABLE_FUNCTION_COLLECTION.setdefault(group, [])
            TABLE_FUNCTION_COLLECTION[group].append(klass)
        return klass
    return decorator