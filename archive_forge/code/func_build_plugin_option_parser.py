import importlib
import logging
import sys
from osc_lib import clientmanager
from osc_lib import shell
import stevedore
def build_plugin_option_parser(parser):
    """Add plugin options to the parser"""
    for mod in PLUGIN_MODULES:
        parser = mod.build_option_parser(parser)
    return parser