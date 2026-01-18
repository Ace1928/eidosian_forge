import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def get_optparser(options):
    """Generate an optparse parser for breezy-style options"""
    parser = OptionParser()
    parser.remove_option('--help')
    for option in options:
        option.add_option(parser, option.short_name())
    return parser