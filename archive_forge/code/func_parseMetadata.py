from twisted import copyright
from twisted.web import http
def parseMetadata(self, data):
    meta = []
    for chunk in data.split(';'):
        chunk = chunk.strip().replace('\x00', '')
        if not chunk:
            continue
        key, value = chunk.split('=', 1)
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        meta.append((key, value))
    return meta