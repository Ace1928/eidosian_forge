import os
import warnings
import re
def findmatch(caps, MIMEtype, key='view', filename='/dev/null', plist=[]):
    """Find a match for a mailcap entry.

    Return a tuple containing the command line, and the mailcap entry
    used; (None, None) if no match is found.  This may invoke the
    'test' command of several matching entries before deciding which
    entry to use.

    """
    if _find_unsafe(filename):
        msg = 'Refusing to use mailcap with filename %r. Use a safe temporary filename.' % (filename,)
        warnings.warn(msg, UnsafeMailcapInput)
        return (None, None)
    entries = lookup(caps, MIMEtype, key)
    for e in entries:
        if 'test' in e:
            test = subst(e['test'], filename, plist)
            if test is None:
                continue
            if test and os.system(test) != 0:
                continue
        command = subst(e[key], MIMEtype, filename, plist)
        if command is not None:
            return (command, e)
    return (None, None)