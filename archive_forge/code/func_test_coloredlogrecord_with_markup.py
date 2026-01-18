import logging
import os
import pathlib
import sys
import time
import pytest
def test_coloredlogrecord_with_markup():
    from kivy.logger import ColoredLogRecord
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.INFO, pathname='test.py', lineno=1, msg='Part1: $BOLDPart2$RESET Part 3', args=('args',), exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = ColoredLogRecord(originallogrecord)
    assert str(shimmedlogrecord) == '<LogRecord: kivy.test, 20, test.py, 1, "Part1: \x1b[1mPart2\x1b[0m Part 3">'
    assert originallogrecord.levelname != shimmedlogrecord.levelname
    assert shimmedlogrecord.levelname == '\x1b[1;32mINFO\x1b[0m'