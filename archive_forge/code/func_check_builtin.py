import sys, os
import textwrap
def check_builtin(option, opt, value):
    cvt, what = _builtin_cvt[option.type]
    try:
        return cvt(value)
    except ValueError:
        raise OptionValueError(_('option %s: invalid %s value: %r') % (opt, what, value))