import getpass
import logging
import sys
import traceback
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from oslo_utils import importutils
from oslo_utils import strutils
from osc_lib.cli import client_config as cloud_config
from osc_lib import clientmanager
from osc_lib.command import timing
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import logs
from osc_lib import utils
from osc_lib import version
def init_profile(self):
    self.do_profile = osprofiler_profiler and self.options.profile
    if self.do_profile:
        osprofiler_profiler.init(self.options.profile)