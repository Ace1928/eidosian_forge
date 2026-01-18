import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
def dump_error_file(self, rootcause_error_file: str, error_code: int=0):
    """Dump parent error file from child process's root cause error and error code."""
    with open(rootcause_error_file) as fp:
        rootcause_error = json.load(fp)
        if error_code:
            self.override_error_code_in_rootcause_data(rootcause_error_file, rootcause_error, error_code)
        log.debug('child error file (%s) contents:\n%s', rootcause_error_file, json.dumps(rootcause_error, indent=2))
    my_error_file = self._get_error_file_path()
    if my_error_file:
        self._rm(my_error_file)
        self._write_error_file(my_error_file, json.dumps(rootcause_error))
        log.info("dumped error file to parent's %s", my_error_file)
    else:
        log.error('no error file defined for parent, to copy child error file (%s)', rootcause_error_file)