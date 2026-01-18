import errno
import logging
import os
import subprocess
import typing
from abc import ABC
from abc import abstractmethod
from io import IOBase
from platform import system
from subprocess import PIPE
from time import sleep
from urllib import request
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import utils
def assert_process_still_running(self) -> None:
    """Check if the underlying process is still running."""
    return_code = self.process.poll()
    if return_code:
        raise WebDriverException(f'Service {self._path} unexpectedly exited. Status code was: {return_code}')