from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
def make_standard_option(*args, **kwargs):
    option = make_option(*args, **kwargs)
    PybtexOption.STANDARD_OPTIONS[option.dest] = option
    return option