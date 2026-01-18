from __future__ import absolute_import, division, print_function
import datetime
import uuid
def is_null_or_empty(name):
    if type(name) is bool:
        return False
    if not name or name == '':
        return True
    return False