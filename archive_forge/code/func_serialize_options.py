import logging
import re
def serialize_options(opts):
    s = ','.join(opts)
    if ' ' in s or '\t' in s:
        return 'opts="' + s + '"'
    return 'opts=' + s