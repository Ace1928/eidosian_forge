from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def iter_commands(self):
    """Iterator returning ImportCommand objects."""
    while True:
        line = self.next_line()
        if line is None:
            if b'done' in self.features:
                raise errors.PrematureEndOfStream(self.lineno)
            break
        elif len(line) == 0 or line.startswith(b'#'):
            continue
        elif line.startswith(b'commit '):
            yield self._parse_commit(line[len(b'commit '):])
        elif line.startswith(b'blob'):
            yield self._parse_blob()
        elif line.startswith(b'done'):
            break
        elif line.startswith(b'progress '):
            yield commands.ProgressCommand(line[len(b'progress '):])
        elif line.startswith(b'reset '):
            yield self._parse_reset(line[len(b'reset '):])
        elif line.startswith(b'tag '):
            yield self._parse_tag(line[len(b'tag '):])
        elif line.startswith(b'checkpoint'):
            yield commands.CheckpointCommand()
        elif line.startswith(b'feature'):
            yield self._parse_feature(line[len(b'feature '):])
        else:
            self.abort(errors.InvalidCommand, line)