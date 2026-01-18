from functools import partial
import json
import math
import warnings
from fiona.model import Geometry, to_dict
from fiona._vendor.munch import munchify
def id_record(rec):
    """Converts a record's id to a blank node id and returns the record."""
    rec['id'] = '_:f%s' % rec['id']
    return rec