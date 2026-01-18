import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def strip_prefix(p):
    return '/'.join(p.split('/')[1:])