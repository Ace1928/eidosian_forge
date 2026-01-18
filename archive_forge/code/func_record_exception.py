import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
def record_exception(self, e: BaseException) -> None:
    """
        Write a structured information about the exception into an error file in JSON format.

        If the error file cannot be determined, then logs the content
        that would have been written to the error file.
        """
    file = self._get_error_file_path()
    if file:
        data = {'message': {'message': f'{type(e).__name__}: {e}', 'extraInfo': {'py_callstack': traceback.format_exc(), 'timestamp': str(int(time.time()))}}}
        with open(file, 'w') as fp:
            json.dump(data, fp)