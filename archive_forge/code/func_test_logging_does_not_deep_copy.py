import logging
import os
import pathlib
import sys
import time
import pytest
def test_logging_does_not_deep_copy():
    import threading
    from kivy.logger import Logger

    class UncopyableDatastructure:

        def __init__(self, name):
            self._lock = threading.Lock()
            self._name = name

        def __str__(self):
            return 'UncopyableDatastructure(name=%r)' % self._name
    s = UncopyableDatastructure('Uncopyable')
    Logger.error('The value of s is %s', s)