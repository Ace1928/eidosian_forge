import os
import re
from paste.fileapp import FileApp
from paste.response import header_value, remove_header
def insert_head(body, text):
    end_head = re.search('</head>', body, re.I)
    if end_head:
        return body[:end_head.start()] + text + body[end_head.end():]
    else:
        return text + body