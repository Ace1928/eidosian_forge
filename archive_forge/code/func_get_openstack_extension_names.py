import os
import re
import shlex
import subprocess
from osc_lib.tests import utils
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
@classmethod
def get_openstack_extension_names(cls):
    opts = cls.get_opts(['Name'])
    return cls.openstack('extension list ' + opts)