import sys, string, re
import getopt
from distutils.errors import *
def getopt(self, args=None, object=None):
    """Parse command-line options in args. Store as attributes on object.

        If 'args' is None or not supplied, uses 'sys.argv[1:]'.  If
        'object' is None or not supplied, creates a new OptionDummy
        object, stores option values there, and returns a tuple (args,
        object).  If 'object' is supplied, it is modified in place and
        'getopt()' just returns 'args'; in both cases, the returned
        'args' is a modified copy of the passed-in 'args' list, which
        is left untouched.
        """
    if args is None:
        args = sys.argv[1:]
    if object is None:
        object = OptionDummy()
        created_object = True
    else:
        created_object = False
    self._grok_option_table()
    short_opts = ' '.join(self.short_opts)
    try:
        opts, args = getopt.getopt(args, short_opts, self.long_opts)
    except getopt.error as msg:
        raise DistutilsArgError(msg)
    for opt, val in opts:
        if len(opt) == 2 and opt[0] == '-':
            opt = self.short2long[opt[1]]
        else:
            assert len(opt) > 2 and opt[:2] == '--'
            opt = opt[2:]
        alias = self.alias.get(opt)
        if alias:
            opt = alias
        if not self.takes_arg[opt]:
            assert val == '', "boolean option can't have value"
            alias = self.negative_alias.get(opt)
            if alias:
                opt = alias
                val = 0
            else:
                val = 1
        attr = self.attr_name[opt]
        if val and self.repeat.get(attr) is not None:
            val = getattr(object, attr, 0) + 1
        setattr(object, attr, val)
        self.option_order.append((opt, val))
    if created_object:
        return (args, object)
    else:
        return args