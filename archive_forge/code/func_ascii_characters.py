import os
import re
import six
from six.moves import urllib
from routes import request_config
def ascii_characters(string):
    if string is None:
        return True
    return all((ord(c) < 128 for c in string))