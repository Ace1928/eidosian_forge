import importlib
import math
import re
from enum import Enum
def detect_email(self, text):
    text = text.lower()
    return re.findall(self.email_regex, text)