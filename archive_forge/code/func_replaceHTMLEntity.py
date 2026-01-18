import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def replaceHTMLEntity(t):
    """Helper parser action to replace common HTML entities with their special characters"""
    return _htmlEntityMap.get(t.entity)