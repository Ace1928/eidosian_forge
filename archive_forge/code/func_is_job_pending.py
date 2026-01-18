import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def is_job_pending(self, task_id):
    task_id = int(task_id)
    if task_id in self._task_dictionary:
        job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
        if job_is_pending:
            self._run_qstat('checking job pending status {0}'.format(task_id), False)
            job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
    else:
        self._run_qstat('checking job pending status {0}'.format(task_id), True)
        if task_id in self._task_dictionary:
            job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
        else:
            sge_debug_print('ERROR: Job {0} not in task list, even after forced qstat!'.format(task_id))
            job_is_pending = False
    if not job_is_pending:
        sge_debug_print('DONE! Returning for {0} claiming done!'.format(task_id))
        if task_id in self._task_dictionary:
            sge_debug_print('NOTE: Adding {0} to OutOfScopeJobs list!'.format(task_id))
            self._out_of_scope_jobs.append(int(task_id))
            self._task_dictionary.pop(task_id)
        else:
            sge_debug_print('ERROR: Job {0} not in task list, but attempted to be removed!'.format(task_id))
    return job_is_pending