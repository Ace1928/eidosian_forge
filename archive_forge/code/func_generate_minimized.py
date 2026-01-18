import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def generate_minimized(self, kargs):
    """Generate a minimized version of the URL"""
    routelist = self.routebackwards
    urllist = []
    gaps = False
    for part in routelist:
        if isinstance(part, dict) and part['type'] in (':', '.'):
            arg = part['name']
            has_arg = arg in kargs
            has_default = arg in self.defaults
            if has_default and (not has_arg) and (not gaps):
                continue
            if (has_default and has_arg) and self.make_unicode(kargs[arg]) == self.make_unicode(self.defaults[arg]) and (not gaps):
                continue
            if has_arg and kargs[arg] is None and has_default and (not gaps):
                continue
            elif has_arg:
                val = kargs[arg]
            elif has_default and self.defaults[arg] is not None:
                val = self.defaults[arg]
            elif part['type'] == '.':
                continue
            else:
                return False
            val = as_unicode(val, self.encoding)
            urllist.append(url_quote(val, self.encoding))
            if part['type'] == '.':
                urllist.append('.')
            if has_arg:
                del kargs[arg]
            gaps = True
        elif isinstance(part, dict) and part['type'] == '*':
            arg = part['name']
            kar = kargs.get(arg)
            if kar is not None:
                urllist.append(url_quote(kar, self.encoding))
                gaps = True
        elif part and part[-1] in self.done_chars:
            if not gaps and part in self.done_chars:
                continue
            elif not gaps:
                urllist.append(part[:-1])
                gaps = True
            else:
                gaps = True
                urllist.append(part)
        else:
            gaps = True
            urllist.append(part)
    urllist.reverse()
    url = ''.join(urllist)
    return url