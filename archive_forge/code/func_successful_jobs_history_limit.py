from pprint import pformat
from six import iteritems
import re
@successful_jobs_history_limit.setter
def successful_jobs_history_limit(self, successful_jobs_history_limit):
    """
        Sets the successful_jobs_history_limit of this V2alpha1CronJobSpec.
        The number of successful finished jobs to retain. This is a pointer to
        distinguish between explicit zero and not specified.

        :param successful_jobs_history_limit: The successful_jobs_history_limit
        of this V2alpha1CronJobSpec.
        :type: int
        """
    self._successful_jobs_history_limit = successful_jobs_history_limit