import re
import sys
def get_int_opt(options, optname, default=None):
    string = options.get(optname, default)
    try:
        return int(string)
    except TypeError:
        raise OptionError('Invalid type %r for option %s; you must give an integer value' % (string, optname))
    except ValueError:
        raise OptionError('Invalid value %r for option %s; you must give an integer value' % (string, optname))