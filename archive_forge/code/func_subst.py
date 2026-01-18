import os
import warnings
import re
def subst(field, MIMEtype, filename, plist=[]):
    res = ''
    i, n = (0, len(field))
    while i < n:
        c = field[i]
        i = i + 1
        if c != '%':
            if c == '\\':
                c = field[i:i + 1]
                i = i + 1
            res = res + c
        else:
            c = field[i]
            i = i + 1
            if c == '%':
                res = res + c
            elif c == 's':
                res = res + filename
            elif c == 't':
                if _find_unsafe(MIMEtype):
                    msg = 'Refusing to substitute MIME type %r into a shell command.' % (MIMEtype,)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + MIMEtype
            elif c == '{':
                start = i
                while i < n and field[i] != '}':
                    i = i + 1
                name = field[start:i]
                i = i + 1
                param = findparam(name, plist)
                if _find_unsafe(param):
                    msg = 'Refusing to substitute parameter %r (%s) into a shell command' % (param, name)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + param
            else:
                res = res + '%' + c
    return res