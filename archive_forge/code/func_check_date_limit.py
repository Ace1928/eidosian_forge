import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo
def check_date_limit(self, data, verbose=False):
    """
        Validate date limits.
        """
    if self.upper_date_limit or self.lower_date_limit:
        date_fmt = '%a %b %d %H:%M:%S +0000 %Y'
        tweet_date = datetime.strptime(data['created_at'], date_fmt).replace(tzinfo=timezone.utc)
        if self.upper_date_limit and tweet_date > self.upper_date_limit or (self.lower_date_limit and tweet_date < self.lower_date_limit):
            if self.upper_date_limit:
                message = 'earlier'
                date_limit = self.upper_date_limit
            else:
                message = 'later'
                date_limit = self.lower_date_limit
            if verbose:
                print('Date limit {} is {} than date of current tweet {}'.format(date_limit, message, tweet_date))
            self.do_stop = True