import typing
from pip._vendor.tenacity import _utils
def after_log(logger: 'logging.Logger', log_level: int, sec_format: str='%0.3f') -> typing.Callable[['RetryCallState'], None]:
    """After call strategy that logs to some logger the finished attempt."""

    def log_it(retry_state: 'RetryCallState') -> None:
        if retry_state.fn is None:
            fn_name = '<unknown>'
        else:
            fn_name = _utils.get_callback_name(retry_state.fn)
        logger.log(log_level, f"Finished call to '{fn_name}' after {sec_format % retry_state.seconds_since_start}(s), this was the {_utils.to_ordinal(retry_state.attempt_number)} time calling it.")
    return log_it