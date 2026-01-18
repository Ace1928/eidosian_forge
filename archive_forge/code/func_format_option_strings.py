import sys, os
import textwrap
def format_option_strings(self, option):
    """Return a comma-separated list of option strings & metavariables."""
    if option.takes_value():
        metavar = option.metavar or option.dest.upper()
        short_opts = [self._short_opt_fmt % (sopt, metavar) for sopt in option._short_opts]
        long_opts = [self._long_opt_fmt % (lopt, metavar) for lopt in option._long_opts]
    else:
        short_opts = option._short_opts
        long_opts = option._long_opts
    if self.short_first:
        opts = short_opts + long_opts
    else:
        opts = long_opts + short_opts
    return ', '.join(opts)