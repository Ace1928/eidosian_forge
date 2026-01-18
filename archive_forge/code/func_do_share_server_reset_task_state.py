from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('share_server_id', metavar='<share_server_id>', help='ID of share server to complete migration.')
@cliutils.arg('--task-state', '--task_state', '--state', metavar='<task_state>', default='None', action='single_alias', required=False, help='Indicate which task state to assign the share server. Options: migration_starting, migration_in_progress, migration_completing, migration_success, migration_error, migration_cancel_in_progress, migration_cancelled, migration_driver_in_progress, migration_driver_phase1_done. If no value is provided, None will be used.')
@api_versions.wraps('2.57')
@api_versions.experimental_api
def do_share_server_reset_task_state(cs, args):
    """Explicitly update the task state of a share

    (Admin only, Experimental).
    """
    state = args.task_state
    if args.task_state == 'None':
        state = None
    share_server = _find_share_server(cs, args.share_server_id)
    share_server.reset_task_state(state)