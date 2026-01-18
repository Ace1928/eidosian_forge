import typing
from pip._vendor.tenacity import _utils
def before_sleep_nothing(retry_state: 'RetryCallState') -> None:
    """Before call strategy that does nothing."""