from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def get_atoms(self):
    return self.atoms