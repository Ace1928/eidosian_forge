from datetime import datetime
import sys
def strip_suffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string