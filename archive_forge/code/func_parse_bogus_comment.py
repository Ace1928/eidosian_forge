import re
import _markupbase
from html import unescape
def parse_bogus_comment(self, i, report=1):
    rawdata = self.rawdata
    assert rawdata[i:i + 2] in ('<!', '</'), 'unexpected call to parse_comment()'
    pos = rawdata.find('>', i + 2)
    if pos == -1:
        return -1
    if report:
        self.handle_comment(rawdata[i + 2:pos])
    return pos + 1