import re
import _markupbase
from html import unescape
def parse_endtag(self, i):
    rawdata = self.rawdata
    assert rawdata[i:i + 2] == '</', 'unexpected call to parse_endtag'
    match = endendtag.search(rawdata, i + 1)
    if not match:
        return -1
    gtpos = match.end()
    match = endtagfind.match(rawdata, i)
    if not match:
        if self.cdata_elem is not None:
            self.handle_data(rawdata[i:gtpos])
            return gtpos
        namematch = tagfind_tolerant.match(rawdata, i + 2)
        if not namematch:
            if rawdata[i:i + 3] == '</>':
                return i + 3
            else:
                return self.parse_bogus_comment(i)
        tagname = namematch.group(1).lower()
        gtpos = rawdata.find('>', namematch.end())
        self.handle_endtag(tagname)
        return gtpos + 1
    elem = match.group(1).lower()
    if self.cdata_elem is not None:
        if elem != self.cdata_elem:
            self.handle_data(rawdata[i:gtpos])
            return gtpos
    self.handle_endtag(elem)
    self.clear_cdata_mode()
    return gtpos