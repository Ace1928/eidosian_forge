import sys
import requests
@classmethod
def from_httplib(cls, message):
    """Read headers from a Python 2 httplib message object."""
    headers = []
    for line in message.headers:
        if line.startswith((' ', '\t')):
            key, value = headers[-1]
            headers[-1] = (key, value + '\r\n' + line.rstrip())
            continue
        key, value = line.split(':', 1)
        headers.append((key, value.strip()))
    return cls(headers)