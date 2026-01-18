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
def register_groups(db, *groups):
    register_aggregate_groups(db, *groups)
    register_table_function_groups(db, *groups)
    register_udf_groups(db, *groups)