from time import time  # noqa: F401
from typing import TYPE_CHECKING, Any, List, Tuple
def print_engine_status(engine: 'ExecutionEngine') -> None:
    print(format_engine_status(engine))