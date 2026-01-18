from pprint import pformat
from six import iteritems
import re
@termination_grace_period_seconds.setter
def termination_grace_period_seconds(self, termination_grace_period_seconds):
    """
        Sets the termination_grace_period_seconds of this V1PodSpec.
        Optional duration in seconds the pod needs to terminate gracefully. May
        be decreased in delete request. Value must be non-negative integer. The
        value zero indicates delete immediately. If this value is nil, the
        default grace period will be used instead. The grace period is the
        duration in seconds after the processes running in the pod are sent a
        termination signal and the time when the processes are forcibly halted
        with a kill signal. Set this value longer than the expected cleanup time
        for your process. Defaults to 30 seconds.

        :param termination_grace_period_seconds: The
        termination_grace_period_seconds of this V1PodSpec.
        :type: int
        """
    self._termination_grace_period_seconds = termination_grace_period_seconds