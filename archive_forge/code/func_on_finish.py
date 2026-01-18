import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo
@abstractmethod
def on_finish(self):
    """
        Actions when the tweet limit has been reached
        """