import json
from typing import Any, List
from uuid import uuid4
from fs.base import FS as FSBase
from triad import assert_or_throw
@property
def latest(self) -> FSBase:
    """latest checkpoint folder

        :raises AssertionError: if there was no checkpoint
        """
    assert_or_throw(len(self) > 0, 'checkpoint history is empty')
    return self._fs.opendir(self._iterations[-1])