import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation
@contextlib.contextmanager
def open_spinner(message: str) -> Generator[SpinnerInterface, None, None]:
    if sys.stdout.isatty() and logger.getEffectiveLevel() <= logging.INFO:
        spinner: SpinnerInterface = InteractiveSpinner(message)
    else:
        spinner = NonInteractiveSpinner(message)
    try:
        with hidden_cursor(sys.stdout):
            yield spinner
    except KeyboardInterrupt:
        spinner.finish('canceled')
        raise
    except Exception:
        spinner.finish('error')
        raise
    else:
        spinner.finish('done')