import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readtag(self, pos):
    """Get the tag corresponding to the given index. Does parameter substitution."""
    if not pos.current().isdigit():
        Trace.error('Function should be f0,...,f9: f' + pos.current())
        return None
    index = int(pos.skipcurrent())
    if 2 + index > len(self.translated):
        Trace.error('Function f' + str(index) + ' is not defined')
        return None
    tag = self.translated[2 + index]
    if not '$' in tag:
        return tag
    for variable in self.params:
        if variable in tag:
            param = self.params[variable]
            if not param.literal:
                Trace.error('Parameters in tag ' + tag + ' should be literal: {' + variable + '!}')
                continue
            if param.literalvalue:
                value = param.literalvalue
            else:
                value = ''
            tag = tag.replace(variable, value)
    return tag