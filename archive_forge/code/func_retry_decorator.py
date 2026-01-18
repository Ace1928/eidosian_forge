import functools
import random
import time
def retry_decorator(func):

    @functools.wraps(func)
    def _retry_wrapper(*args, **kwargs):
        partial_func = functools.partial(func, *args, **kwargs)
        return _retry_func(func=partial_func, sleep_time_generator=sleep_time_generator, retries=retries, catch_extra_error_codes=catch_extra_error_codes, found_f=found, status_code_from_except_f=status_code_from_exception, base_class=cls.base_class)
    return _retry_wrapper