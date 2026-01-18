from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
def standard_option(name):
    return PybtexOption.STANDARD_OPTIONS[name]