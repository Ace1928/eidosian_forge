import warnings
from typing import NoReturn, Set
from modin.logging import get_logger
from modin.utils import get_current_execution
@classmethod
def non_verified_udf(cls) -> None:
    get_logger().debug('Modin Warning: Non Verified UDF')
    warnings.warn('User-defined function verification is still under development in Modin. ' + 'The function provided is not verified.')