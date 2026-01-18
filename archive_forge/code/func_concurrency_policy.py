from pprint import pformat
from six import iteritems
import re
@concurrency_policy.setter
def concurrency_policy(self, concurrency_policy):
    """
        Sets the concurrency_policy of this V2alpha1CronJobSpec.
        Specifies how to treat concurrent executions of a Job. Valid values are:
        - "Allow" (default): allows CronJobs to run concurrently; -
        "Forbid": forbids concurrent runs, skipping next run if previous run
        hasn't finished yet; - "Replace": cancels currently running job and
        replaces it with a new one

        :param concurrency_policy: The concurrency_policy of this
        V2alpha1CronJobSpec.
        :type: str
        """
    self._concurrency_policy = concurrency_policy