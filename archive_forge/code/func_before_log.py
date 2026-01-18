import typing
from pip._vendor.tenacity import _utils
def before_log(logger: 'logging.Logger', log_level: int) -> typing.Callable[['RetryCallState'], None]:
    """Before call strategy that logs to some logger the attempt."""

    def log_it(retry_state: 'RetryCallState') -> None:
        if retry_state.fn is None:
            fn_name = '<unknown>'
        else:
            fn_name = _utils.get_callback_name(retry_state.fn)
        logger.log(log_level, f"Starting call to '{fn_name}', this is the {_utils.to_ordinal(retry_state.attempt_number)} time calling it.")
    return log_it