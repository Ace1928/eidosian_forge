import logging
import os
import pathlib
import sys
import time
import pytest
def test_colonsplittinglogrecord_without_colon():
    from kivy.logger import ColonSplittingLogRecord
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.DEBUG, pathname='test.py', lineno=1, msg='Part1 Part2 Part 3', args=('args',), exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = ColonSplittingLogRecord(originallogrecord)
    assert str(originallogrecord) == str(shimmedlogrecord)
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.DEBUG, pathname='test.py', lineno=1, msg=1, args=None, exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = ColonSplittingLogRecord(originallogrecord)
    assert str(originallogrecord) == str(shimmedlogrecord)