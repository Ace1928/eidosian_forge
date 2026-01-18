import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def buildnextreg(self, path, clist, include_names=True):
    """Recursively build our regexp given a path, and a controller
        list.

        Returns the regular expression string, and two booleans that
        can be ignored as they're only used internally by buildnextreg.

        """
    if path:
        part = path[0]
    else:
        part = ''
    reg = ''
    rest, noreqs, allblank = ('', True, True)
    if len(path[1:]) > 0:
        self.prior = part
        rest, noreqs, allblank = self.buildnextreg(path[1:], clist, include_names)
    if isinstance(part, dict) and part['type'] in (':', '.'):
        var = part['name']
        typ = part['type']
        partreg = ''
        if var in self.reqs:
            if include_names:
                partreg = '(?P<%s>%s)' % (var, self.reqs[var])
            else:
                partreg = '(?:%s)' % self.reqs[var]
            if typ == '.':
                partreg = '(?:\\.%s)??' % partreg
        elif var == 'controller':
            if include_names:
                partreg = '(?P<%s>%s)' % (var, '|'.join(map(re.escape, clist)))
            else:
                partreg = '(?:%s)' % '|'.join(map(re.escape, clist))
        elif self.prior in ['/', '#']:
            if include_names:
                partreg = '(?P<' + var + '>[^' + self.prior + ']+?)'
            else:
                partreg = '(?:[^' + self.prior + ']+?)'
        elif not rest:
            if typ == '.':
                exclude_chars = '/.'
            else:
                exclude_chars = '/'
            if include_names:
                partreg = '(?P<%s>[^%s]+?)' % (var, exclude_chars)
            else:
                partreg = '(?:[^%s]+?)' % exclude_chars
            if typ == '.':
                partreg = '(?:\\.%s)??' % partreg
        else:
            end = ''.join(self.done_chars)
            rem = rest
            if rem[0] == '\\' and len(rem) > 1:
                rem = rem[1]
            elif rem.startswith('(\\') and len(rem) > 2:
                rem = rem[2]
            else:
                rem = end
            rem = frozenset(rem) | frozenset(['/'])
            if include_names:
                partreg = '(?P<%s>[^%s]+?)' % (var, ''.join(rem))
            else:
                partreg = '(?:[^%s]+?)' % ''.join(rem)
        if var in self.reqs:
            noreqs = False
        if var not in self.defaults:
            allblank = False
            noreqs = False
        if noreqs:
            if var in self.reqs and var in self.defaults:
                reg = '(?:' + partreg + rest + ')?'
            elif var in self.reqs:
                allblank = False
                reg = partreg + rest
            elif var in self.defaults and self.prior in (',', ';', '.'):
                reg = partreg + rest
            elif var in self.defaults:
                reg = partreg + '?' + rest
            else:
                allblank = False
                reg = partreg + rest
        elif allblank and var in self.defaults:
            reg = '(?:' + partreg + rest + ')?'
        else:
            reg = partreg + rest
    elif isinstance(part, dict) and part['type'] == '*':
        var = part['name']
        if noreqs:
            if include_names:
                reg = '(?P<%s>.*)' % var + rest
            else:
                reg = '(?:.*)' + rest
            if var not in self.defaults:
                allblank = False
                noreqs = False
        elif allblank and var in self.defaults:
            if include_names:
                reg = '(?P<%s>.*)' % var + rest
            else:
                reg = '(?:.*)' + rest
        elif var in self.defaults:
            if include_names:
                reg = '(?P<%s>.*)' % var + rest
            else:
                reg = '(?:.*)' + rest
        else:
            if include_names:
                reg = '(?P<%s>.*)' % var + rest
            else:
                reg = '(?:.*)' + rest
            allblank = False
            noreqs = False
    elif part and part[-1] in self.done_chars:
        if allblank:
            reg = re.escape(part[:-1]) + '(?:' + re.escape(part[-1]) + rest
            reg += ')?'
        else:
            allblank = False
            if part == '/':
                reg = '\\/' + rest
            else:
                reg = re.escape(part) + rest
    else:
        noreqs = False
        allblank = False
        reg = re.escape(part) + rest
    return (reg, noreqs, allblank)