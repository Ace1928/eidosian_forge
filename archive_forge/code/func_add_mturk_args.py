import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_mturk_args(self):
    """
        Add standard mechanical turk arguments.
        """
    mturk = self.add_argument_group('Mechanical Turk')
    default_log_path = os.path.join(self.parlai_home, 'logs', 'mturk')
    mturk.add_argument('--mturk-log-path', default=default_log_path, help='path to MTurk logs, defaults to {parlai_dir}/logs/mturk')
    mturk.add_argument('-t', '--task', help='MTurk task, e.g. "qa_data_collection" or "model_evaluator"')
    mturk.add_argument('-nc', '--num-conversations', default=1, type=int, help='number of conversations you want to create for this task')
    mturk.add_argument('--unique', dest='unique_worker', default=False, action='store_true', help='enforce that no worker can work on your task twice')
    mturk.add_argument('--max-hits-per-worker', dest='max_hits_per_worker', default=0, type=int, help='Max number of hits each worker can perform during current group run')
    mturk.add_argument('--unique-qual-name', dest='unique_qual_name', default=None, type=str, help='qualification name to use for uniqueness between HITs')
    mturk.add_argument('-r', '--reward', default=0.05, type=float, help='reward for each worker for finishing the conversation, in US dollars')
    mturk.add_argument('--sandbox', dest='is_sandbox', action='store_true', help='submit the HITs to MTurk sandbox site')
    mturk.add_argument('--live', dest='is_sandbox', action='store_false', help='submit the HITs to MTurk live site')
    mturk.add_argument('--debug', dest='is_debug', action='store_true', help='print and log all server interactions and messages')
    mturk.add_argument('--verbose', dest='verbose', action='store_true', help='print all messages sent to and from Turkers')
    mturk.add_argument('--hard-block', dest='hard_block', action='store_true', default=False, help='Hard block disconnecting Turkers from all of your HITs')
    mturk.add_argument('--log-level', dest='log_level', type=int, default=20, help='importance level for what to put into the logs. the lower the level the more that gets logged. values are 0-50')
    mturk.add_argument('--disconnect-qualification', dest='disconnect_qualification', default=None, help='Qualification to use for soft blocking users for disconnects. By default turkers are never blocked, though setting this will allow you to filter out turkers that have disconnected too many times on previous HITs where this qualification was set.')
    mturk.add_argument('--block-qualification', dest='block_qualification', default=None, help='Qualification to use for soft blocking users. This qualification is granted whenever soft_block_worker is called, and can thus be used to filter workers out from a single task or group of tasks by noted performance.')
    mturk.add_argument('--count-complete', dest='count_complete', default=False, action='store_true', help='continue until the requested number of conversations are completed rather than attempted')
    mturk.add_argument('--allowed-conversations', dest='allowed_conversations', default=0, type=int, help='number of concurrent conversations that one mturk worker is able to be involved in, 0 is unlimited')
    mturk.add_argument('--max-connections', dest='max_connections', default=30, type=int, help='number of HITs that can be launched at the same time, 0 is unlimited.')
    mturk.add_argument('--min-messages', dest='min_messages', default=0, type=int, help='number of messages required to be sent by MTurk agent when considering whether to approve a HIT in the event of a partner disconnect. I.e. if the number of messages exceeds this number, the turker can submit the HIT.')
    mturk.add_argument('--local', dest='local', default=False, action='store_true', help='Run the server locally on this server rather than setting up a heroku server.')
    mturk.add_argument('--hobby', dest='hobby', default=False, action='store_true', help='Run the heroku server on the hobby tier.')
    mturk.add_argument('--max-time', dest='max_time', default=0, type=int, help='Maximum number of seconds per day that a worker is allowed to work on this assignment')
    mturk.add_argument('--max-time-qual', dest='max_time_qual', default=None, help='Qualification to use to share the maximum time requirement with other runs from other machines.')
    mturk.add_argument('--heroku-team', dest='heroku_team', default=None, help='Specify Heroku team name to use for launching Dynos.')
    mturk.add_argument('--tmp-dir', dest='tmp_dir', default=None, help='Specify location to use for scratch builds and such.')
    mturk.set_defaults(interactive_mode=True)
    mturk.set_defaults(is_sandbox=True)
    mturk.set_defaults(is_debug=False)
    mturk.set_defaults(verbose=False)