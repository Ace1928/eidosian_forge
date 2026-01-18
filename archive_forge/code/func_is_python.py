import collections
import copy
import json
import re
import shutil
import tempfile
def is_python(cell):
    """Checks if the cell consists of Python code."""
    return cell['cell_type'] == 'code' and cell['source'] and (not cell['source'][0].startswith('%%'))