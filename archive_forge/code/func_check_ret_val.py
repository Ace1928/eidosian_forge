import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def check_ret_val(self, ret_val, job_path, success_values=[0]):
    """Checks that the job represented by the given arguments succeeded.

        Some Hyper-V operations are not atomic, and will return a reference
        to a job. In this case, this method will wait for the job's
        completion.

        :param ret_val: integer, representing the return value of the job.
            if the value is WMI_JOB_STATUS_STARTED or WMI_JOB_STATE_RUNNING,
            a job_path cannot be None.
        :param job_path: string representing the WMI object path of a
            Hyper-V job.
        :param success_values: list of return values that can be considered
            successful. WMI_JOB_STATUS_STARTED and WMI_JOB_STATE_RUNNING
            values are ignored.
        :raises exceptions.WMIJobFailed: if the given ret_val is
            WMI_JOB_STATUS_STARTED or WMI_JOB_STATE_RUNNING and the state of
            job represented by the given job_path is not
            WMI_JOB_STATE_COMPLETED or JOB_STATE_COMPLETED_WITH_WARNINGS, or
            if the given ret_val is not in the list of given success_values.
        """
    if ret_val in [constants.WMI_JOB_STATUS_STARTED, constants.WMI_JOB_STATE_RUNNING]:
        return self._wait_for_job(job_path)
    elif ret_val not in success_values:
        raise exceptions.WMIJobFailed(error_code=ret_val, job_state=None, error_summ_desc=None, error_desc=None)