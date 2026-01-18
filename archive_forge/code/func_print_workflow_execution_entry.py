import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
def print_workflow_execution_entry(self, wf_ex, level):
    self.print_line("workflow '%s' [%s] %s" % (wf_ex['name'], wf_ex['state'], wf_ex['id']), level)
    if 'task_executions' in wf_ex:
        for t_ex in wf_ex['task_executions']:
            self.print_task_execution_entry(t_ex, level + 1)