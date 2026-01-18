import datetime
import itertools
import os
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from multiprocessing import Process
from typing import Any, Collection, Dict, NoReturn, Optional, Union, cast, overload
from .compat import Literal
from .tables import row, table
from .util import COLORS, ICONS, MESSAGES, can_render
from .util import color as _color
from .util import locale_escape, supports_ansi, wrap
@contextmanager
def loading(self, text: str='Loading...'):
    if self.no_print:
        yield
    elif self.hide_animation:
        print(text)
        yield
    else:
        sys.stdout.flush()
        t = Process(target=self._spinner, args=(text,))
        t.start()
        try:
            yield
        except Exception as e:
            t.terminate()
            sys.stdout.write('\n')
            raise e
        t.terminate()
        sys.stdout.write('\r\x1b[2K')
        sys.stdout.flush()