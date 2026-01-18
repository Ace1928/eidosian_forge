from pprint import pformat
from six import iteritems
import re
@last_schedule_time.setter
def last_schedule_time(self, last_schedule_time):
    """
        Sets the last_schedule_time of this V2alpha1CronJobStatus.
        Information when was the last time the job was successfully scheduled.

        :param last_schedule_time: The last_schedule_time of this
        V2alpha1CronJobStatus.
        :type: datetime
        """
    self._last_schedule_time = last_schedule_time