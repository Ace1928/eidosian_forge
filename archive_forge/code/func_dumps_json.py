import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def dumps_json(self):
    return json.dumps(self.data)