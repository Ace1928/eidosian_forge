import re
import _markupbase
from html import unescape
def check_for_whole_start_tag(self, i):
    rawdata = self.rawdata
    m = locatestarttagend_tolerant.match(rawdata, i)
    if m:
        j = m.end()
        next = rawdata[j:j + 1]
        if next == '>':
            return j + 1
        if next == '/':
            if rawdata.startswith('/>', j):
                return j + 2
            if rawdata.startswith('/', j):
                return -1
            if j > i:
                return j
            else:
                return i + 1
        if next == '':
            return -1
        if next in 'abcdefghijklmnopqrstuvwxyz=/ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return -1
        if j > i:
            return j
        else:
            return i + 1
    raise AssertionError('we should not get here!')