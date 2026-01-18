from functools import partial
import json
import math
import warnings
from fiona.model import Geometry, to_dict
from fiona._vendor.munch import munchify
def obj_gen(lines, object_hook=None):
    """Return a generator of JSON objects loaded from ``lines``."""
    first_line = next(lines)
    if first_line.startswith('\x1e'):

        def gen():
            buffer = first_line.strip('\x1e')
            for line in lines:
                if line.startswith('\x1e'):
                    if buffer:
                        yield json.loads(buffer, object_hook=object_hook)
                    buffer = line.strip('\x1e')
                else:
                    buffer += line
            else:
                yield json.loads(buffer, object_hook=object_hook)
    else:

        def gen():
            yield json.loads(first_line, object_hook=object_hook)
            for line in lines:
                yield json.loads(line, object_hook=object_hook)
    return gen()