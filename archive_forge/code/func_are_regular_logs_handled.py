import logging
import os
import pathlib
import sys
import time
import pytest
def are_regular_logs_handled():
    from kivy.logger import LoggerHistory
    LoggerHistory.clear_history()
    logging.getLogger('test').info(1)
    return bool(LoggerHistory.history)