from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def id_formatter(label: str):
    """
    Return a function that formats the value given to it with the given label.
    """
    return lambda value: format_with_label(label, value)