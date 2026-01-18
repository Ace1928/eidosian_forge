import logging
import os
import pathlib
import sys
import time
import pytest
def test_kivyformatter_colon_no_color():
    from kivy.logger import KivyFormatter
    formatter = KivyFormatter('[%(levelname)-7s] %(message)s', use_color=False)
    logger, log_output = configured_string_logging('1', formatter)
    logger.info('Fancy: $BOLDmess$RESETage')
    assert log_output.getvalue() == '[INFO   ] [Fancy       ] message\n'