import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def help_add(self):
    print('Add profile info from given file to current statistics object.', file=self.stream)