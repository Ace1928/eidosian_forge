import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
def print_action_execution_entry(self, a_ex, level):
    self.print_line("action '%s' [%s] %s" % (a_ex['name'], a_ex['state'], a_ex['id']), level)
    if a_ex['state'] == 'ERROR':
        state_info = a_ex['state_info']
        if state_info:
            state_info = state_info[0:100] + '...'
            self.print_line('(error info: %s)' % state_info, level)