import logging
import os
import pathlib
import sys
import time
import pytest
def are_kivy_logger_logs_handled():
    from kivy.logger import LoggerHistory, Logger
    LoggerHistory.clear_history()
    Logger.info(1)
    return bool(LoggerHistory.history)