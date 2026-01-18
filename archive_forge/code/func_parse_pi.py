import re
import _markupbase
from html import unescape
def parse_pi(self, i):
    rawdata = self.rawdata
    assert rawdata[i:i + 2] == '<?', 'unexpected call to parse_pi()'
    match = piclose.search(rawdata, i + 2)
    if not match:
        return -1
    j = match.start()
    self.handle_pi(rawdata[i + 2:j])
    j = match.end()
    return j