import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def onclick_goto__get(self):
    return 'location.href=%s; return false' % js_repr(self.href)