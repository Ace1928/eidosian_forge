from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from modin.config import LogMode
from .config import LogLevel, get_logger
@wraps(obj)
def run_and_log(*args: Tuple, **kwargs: Dict) -> Any:
    """
            Compute function with logging if Modin logging is enabled.

            Parameters
            ----------
            *args : tuple
                The function arguments.
            **kwargs : dict
                The function keyword arguments.

            Returns
            -------
            Any
            """
    if LogMode.get() == 'disable':
        return obj(*args, **kwargs)
    logger = get_logger()
    logger.log(log_level, start_line)
    try:
        result = obj(*args, **kwargs)
    except BaseException as e:
        if not hasattr(e, '_modin_logged'):
            get_logger('modin.logger.errors').exception(stop_line, stack_info=True)
            e._modin_logged = True
        raise
    finally:
        logger.log(log_level, stop_line)
    return result