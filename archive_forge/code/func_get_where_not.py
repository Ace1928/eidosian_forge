import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def get_where_not(jdata, *args, **kwargs):
    if isinstance(jdata, dict):
        jdata = [jdata]
    match = []
    for entry in jdata:
        match_args = all([arg in entry.keys() or arg in entry.values() for arg in args])
        match_kwargs = all([entry[key] == kwargs[key] for key in kwargs.keys()])
        if not match_args and (not match_kwargs):
            match.append(entry)
    return match