import os
import warnings
import re
def parseline(line):
    """Parse one entry in a mailcap file and return a dictionary.

    The viewing command is stored as the value with the key "view",
    and the rest of the fields produce key-value pairs in the dict.
    """
    fields = []
    i, n = (0, len(line))
    while i < n:
        field, i = parsefield(line, i, n)
        fields.append(field)
        i = i + 1
    if len(fields) < 2:
        return (None, None)
    key, view, rest = (fields[0], fields[1], fields[2:])
    fields = {'view': view}
    for field in rest:
        i = field.find('=')
        if i < 0:
            fkey = field
            fvalue = ''
        else:
            fkey = field[:i].strip()
            fvalue = field[i + 1:].strip()
        if fkey in fields:
            pass
        else:
            fields[fkey] = fvalue
    return (key, fields)