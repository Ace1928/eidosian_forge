import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
@property
def mon_version(self):
    return self.app_args.monasca_api_version