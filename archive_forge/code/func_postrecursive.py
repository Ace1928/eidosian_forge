import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def postrecursive(self, container):
    """Postprocess the container contents recursively"""
    if not hasattr(container, 'contents'):
        return
    if len(container.contents) == 0:
        return
    if hasattr(container, 'postprocess'):
        if not container.postprocess:
            return
    postprocessor = Postprocessor()
    contents = []
    for element in container.contents:
        post = postprocessor.postprocess(element)
        if post:
            contents.append(post)
    for i in range(2):
        post = postprocessor.postprocess(None)
        if post:
            contents.append(post)
    container.contents = contents