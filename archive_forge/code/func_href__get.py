import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def href__get(self):
    s = self.url
    if self.vars:
        s += '?'
        vars = []
        for name, val in self.vars:
            if isinstance(val, (list, tuple)):
                val = [v for v in val if v is not None]
            elif val is None:
                continue
            vars.append((name, val))
        s += urlencode(vars, True)
    return s