import importlib
import math
import re
from enum import Enum
def detect_phone_number(self, text):
    phones = re.findall(self.phone_number_regex, text)
    edited = []
    for tup in phones:
        edited.append(''.join(list(tup)))
    return edited