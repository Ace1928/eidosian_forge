import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
def decode_unicode(coll):
    if isinstance(coll, list):
        for no, entry in enumerate(coll):
            if isinstance(entry, (dict, list)):
                coll[no] = decode_unicode(entry)
            elif isinstance(entry, str):
                try:
                    coll[no] = str(entry)
                except UnicodeEncodeError:
                    pass
    elif isinstance(coll, dict):
        keys, vals = (list(coll.keys()), list(coll.values()))
        for key, val in zip(keys, vals):
            if isinstance(val, (dict, list)):
                coll[key] = decode_unicode(val)
            elif isinstance(val, str):
                try:
                    coll[key] = str(val)
                except UnicodeEncodeError:
                    pass
            coll[str(key)] = coll.pop(key)
    return coll