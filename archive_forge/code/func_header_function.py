import pycurl
import re
def header_function(header_line):
    header_line = header_line.decode('iso-8859-1')
    if ':' not in header_line:
        return
    name, value = header_line.split(':', 1)
    name = name.strip()
    value = value.strip()
    name = name.lower()
    headers[name] = value