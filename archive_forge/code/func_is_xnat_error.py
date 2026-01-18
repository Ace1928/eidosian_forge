import re
from lxml import etree
def is_xnat_error(message):
    a = ['<!DOCTYPE', '<html>']
    try:
        return message.startswith(a[0]) or message.startswith(a[1])
    except TypeError:
        if isinstance(message, bytes):
            a = [bytes(e, 'utf-8') for e in a]
        return message.startswith(a[0]) or message.startswith(a[1])