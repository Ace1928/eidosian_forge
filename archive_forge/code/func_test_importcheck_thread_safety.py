from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
@pytest.mark.slow
@pytest.mark.single_cpu
def test_importcheck_thread_safety(self, datapath, flavor_read_html):

    class ErrorThread(threading.Thread):

        def run(self):
            try:
                super().run()
            except Exception as err:
                self.err = err
            else:
                self.err = None
    filename = datapath('io', 'data', 'html', 'valid_markup.html')
    helper_thread1 = ErrorThread(target=flavor_read_html, args=(filename,))
    helper_thread2 = ErrorThread(target=flavor_read_html, args=(filename,))
    helper_thread1.start()
    helper_thread2.start()
    while helper_thread1.is_alive() or helper_thread2.is_alive():
        pass
    assert None is helper_thread1.err is helper_thread2.err