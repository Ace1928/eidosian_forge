from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_filenotfound(exc: FileNotFoundError) -> bool:
    return not exc.filename.endswith('.tmp')