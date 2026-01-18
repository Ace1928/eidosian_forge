import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def create_error_func(message):

    def func(*args):
        raise ValueError(message % file_attr)
    return func