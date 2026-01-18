import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def csv_to_json(csv_str):
    import csv
    try:
        csv_reader = csv.reader(StringIO(csv_str), delimiter=',', quotechar='"')
    except TypeError:
        csv_reader = csv.reader(StringIO(csv_str.decode('utf-8')), delimiter=',', quotechar='"')
    headers = next(csv_reader)
    ans = [dict(zip(headers, entry)) for entry in csv_reader]
    return ans