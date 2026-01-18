import warnings
from typing import NoReturn, Set
from modin.logging import get_logger
from modin.utils import get_current_execution
@classmethod
def not_initialized(cls, engine: str, code: str) -> None:
    get_logger().debug(f'Modin Warning: Not Initialized: {engine}')
    warnings.warn(f'{engine} execution environment not yet initialized. Initializing...\n' + 'To remove this warning, run the following python code before doing dataframe operations:\n' + f'{code}')