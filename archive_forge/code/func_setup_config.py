import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
def setup_config(conf):
    config_struct = config.Config(TEST_CONFIG)
    for key, value in conf.items():
        if not hasattr(config_struct, key):
            raise ValueError(f'{key!r} is not a valid config attribute')
        setattr(config_struct, key, value)
    return config_struct