from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def escapeattr(data):
    data = escape(data)
    data = data.replace('"', '&quot;')
    return data