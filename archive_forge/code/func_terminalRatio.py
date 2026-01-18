import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
@property
def terminalRatio(self):
    """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
    return self.terms_count / float(self.terms_count + self.prims_count)