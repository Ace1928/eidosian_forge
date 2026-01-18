import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def generate_json(self):
    """Call the OpenAI API to generate a JSON object."""
    raise NotImplementedError