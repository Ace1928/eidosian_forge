import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec
def signature_from_docstring(doc, obj_name):
    args = '()'
    try:
        found = False
        if len(doc) > 0:
            if IS_IPY:
                if obj_name:
                    name = obj_name + '('
                    lines = doc.splitlines()
                    if len(lines) == 1:
                        c = doc.count(name)
                        if c > 1:
                            doc = ('\n' + name).join(doc.split(name))
                    major = ''
                    for line in doc.splitlines():
                        if line.startswith(name) and line.endswith(')'):
                            if len(line) > len(major):
                                major = line
                    if major:
                        args = major[major.index('('):]
                        found = True
            if not found:
                i = doc.find('->')
                if i < 0:
                    i = doc.find('--')
                    if i < 0:
                        i = doc.find('\n')
                        if i < 0:
                            i = doc.find('\r')
                if i > 0:
                    s = doc[0:i]
                    s = s.strip()
                    if s[-1] == ')':
                        start = s.find('(')
                        if start >= 0:
                            end = s.find('[')
                            if end <= 0:
                                end = s.find(')')
                                if end <= 0:
                                    end = len(s)
                            args = s[start:end]
                            if not args[-1] == ')':
                                args = args + ')'
                            l = len(args) - 1
                            r = []
                            for i in range(len(args)):
                                if i == 0 or i == l:
                                    r.append(args[i])
                                else:
                                    r.append(check_char(args[i]))
                            args = ''.join(r)
            if IS_IPY:
                if args.startswith('(self:'):
                    i = args.find(',')
                    if i >= 0:
                        args = '(self' + args[i:]
                    else:
                        args = '(self)'
                i = args.find(')')
                if i > 0:
                    args = args[:i + 1]
    except:
        pass
    return (args, doc)