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
def close_profile(self):
    if self.do_profile:
        profiler = osprofiler_profiler.get()
        trace_id = profiler.get_base_id()
        short_id = profiler.get_shorten_id(trace_id)
        self.log.warning('Trace ID: %s' % trace_id)
        self.log.warning('Short trace ID for OpenTracing-based drivers: %s' % short_id)
        self.log.warning('Display trace data with command:\nosprofiler trace show --html %s ' % trace_id)