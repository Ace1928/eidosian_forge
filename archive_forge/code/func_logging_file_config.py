import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def logging_file_config(self, config_file):
    """
        Setup logging via the logging module's fileConfig function with the
        specified ``config_file``, if applicable.

        ConfigParser defaults are specified for the special ``__file__``
        and ``here`` variables, similar to PasteDeploy config loading.
        """
    parser = configparser.ConfigParser()
    parser.read([config_file])
    if parser.has_section('loggers'):
        config_file = os.path.abspath(config_file)
        fileConfig(config_file, dict(__file__=config_file, here=os.path.dirname(config_file)))