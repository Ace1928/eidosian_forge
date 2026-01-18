import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def no_setting_conf_directly_in_tests(logical_line, filename):
    """Check for setting CONF.* attributes directly in tests

    The value can leak out of tests affecting how subsequent tests run.
    Using self.flags(option=value) is the preferred method to temporarily
    set config options in tests.

    N320
    """
    if 'os_win/tests/' in filename:
        res = conf_attribute_set_re.match(logical_line)
        if res:
            yield (0, 'N320: Setting CONF.* attributes directly in tests is forbidden. Use self.flags(option=value) instead')