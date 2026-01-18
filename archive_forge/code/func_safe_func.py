import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional
def safe_func(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        _, _, exc_traceback = sys.exc_info()
        traceback_details = traceback.extract_tb(exc_traceback)
        filename = traceback_details[-1].filename
        lineno = traceback_details[-1].lineno
        logger.debug(f'Exception: func={func!r} args={args!r} kwargs={kwargs!r} e={e!r} filename={filename!r} lineno={lineno!r}. traceback_details={traceback_details!r}')
        if raise_on_error:
            raise e