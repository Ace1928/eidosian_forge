import functools
import random
import time
@classmethod
def jittered_backoff(cls, retries=10, delay=3, backoff=2.0, max_delay=60, catch_extra_error_codes=None):
    """Wrap a callable with retry behavior.
        Args:
            retries (int): Number of times to retry a failed request before giving up
                default=10
            delay (int or float): Initial delay between retries in seconds
                default=3
            backoff (int or float): backoff multiplier e.g. value of 2 will  double the delay each retry
                default=2.0
            max_delay (int or None): maximum amount of time to wait between retries.
                default=60
            catch_extra_error_codes: Additional error messages to catch, in addition to those which may be defined by a subclass of CloudRetry
                default=None
        Returns:
            Callable: A generator that calls the decorated function using using a jittered backoff strategy.
        """
    sleep_time_generator = BackoffIterator(delay=delay, backoff=backoff, max_delay=max_delay, jitter=True)
    return cls.base_decorator(retries=retries, found=cls.found, status_code_from_exception=cls.status_code_from_exception, catch_extra_error_codes=catch_extra_error_codes, sleep_time_generator=sleep_time_generator)