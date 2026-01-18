import importlib
import math
import re
from enum import Enum
def detect_ssn(self, text):
    return re.findall(self.ssn_regex, text)