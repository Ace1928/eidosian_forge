import time
from datetime import datetime
from sys import version_info
def strptime(text, dateFormat):
    return datetime.strptime(text, dateFormat)