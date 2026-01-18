import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
@classmethod
def assertInOutput(cls, expected, actual):
    if expected not in actual:
        raise Exception(expected + ' not in ' + actual)