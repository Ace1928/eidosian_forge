import re
import _markupbase
from html import unescape
def parse_html_declaration(self, i):
    rawdata = self.rawdata
    assert rawdata[i:i + 2] == '<!', 'unexpected call to parse_html_declaration()'
    if rawdata[i:i + 4] == '<!--':
        return self.parse_comment(i)
    elif rawdata[i:i + 3] == '<![':
        return self.parse_marked_section(i)
    elif rawdata[i:i + 9].lower() == '<!doctype':
        gtpos = rawdata.find('>', i + 9)
        if gtpos == -1:
            return -1
        self.handle_decl(rawdata[i + 2:gtpos])
        return gtpos + 1
    else:
        return self.parse_bogus_comment(i)