import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readoption(self, args):
    """Read the key and value for an option"""
    arg = args[0][2:]
    del args[0]
    if '=' in arg:
        key = self.readequalskey(arg, args)
    else:
        key = arg.replace('-', '')
    if not hasattr(self.options, key):
        return (None, key)
    current = getattr(self.options, key)
    if isinstance(current, bool):
        return (key, True)
    if len(args) == 0:
        return (key, None)
    if args[0].startswith('"'):
        initial = args[0]
        del args[0]
        return (key, self.readquoted(args, initial))
    value = args[0].decode('utf-8')
    del args[0]
    if isinstance(current, list):
        current.append(value)
        return (key, current)
    return (key, value)