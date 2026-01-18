from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
def check_plugin(option, option_string, value):
    return find_plugin(option.plugin_group, value)