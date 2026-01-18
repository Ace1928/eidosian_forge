from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
@property
def run_info(self):
    return dict(ignore_value_none=self.ignore_value_none, check_rc=self.check_rc, environ_update=self.environ_update, args_order=self.args_order, cmd=self.cmd, run_command_args=self.run_command_args, context_run_args=self.context_run_args, results_rc=self.results_rc, results_out=self.results_out, results_err=self.results_err, results_processed=self.results_processed)