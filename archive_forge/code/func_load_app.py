import pkg_resources
import argparse
import logging
import sys
from warnings import warn
def load_app(self):
    from pecan import load_app
    return load_app(self.args.config_file)