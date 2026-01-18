import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def stats_str(self):
    """Return a string of patch statistics"""
    return '%i inserts, %i removes in %i hunks' % self.stats_values()