import re
import sys
def get_choice_opt(options, optname, allowed, default=None, normcase=False):
    string = options.get(optname, default)
    if normcase:
        string = string.lower()
    if string not in allowed:
        raise OptionError('Value for option %s must be one of %s' % (optname, ', '.join(map(str, allowed))))
    return string