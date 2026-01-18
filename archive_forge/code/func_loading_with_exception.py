import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def loading_with_exception():
    p = Printer()
    print('\n')
    with p.loading():
        raise Exception('This is an error.')