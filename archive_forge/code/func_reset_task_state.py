import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def reset_task_state(self, share_id, state, version=None):
    state = '--task_state %s' % state if state else ''
    return self.manila('reset-task-state %(state)s %(share)s' % {'state': state, 'share': share_id}, microversion=version)