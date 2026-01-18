import datetime
import json
import numpy as np
from ase.utils import reader, writer
def intkey(key):
    """Convert str to int if possible."""
    try:
        return int(key)
    except ValueError:
        return key