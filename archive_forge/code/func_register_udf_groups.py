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
def register_udf_groups(db, *groups):
    seen = set()
    for group in groups:
        functions = UDF_COLLECTION.get(group, ())
        for function in functions:
            name = function.__name__
            if name not in seen:
                seen.add(name)
                db.register_function(function, name)