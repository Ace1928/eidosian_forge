import warnings
from typing import NoReturn, Set
from modin.logging import get_logger
from modin.utils import get_current_execution
@classmethod
def single_warning(cls, message: str) -> None:
    message_hash = hash(message)
    logger = get_logger()
    if message_hash in cls.printed_warnings:
        logger.debug(f'Modin Warning: Single Warning: {message} was raised and suppressed.')
        return
    logger.debug(f'Modin Warning: Single Warning: {message} was raised.')
    warnings.warn(message)
    cls.printed_warnings.add(message_hash)