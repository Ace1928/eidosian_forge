import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def embed_comments(self, idx, currentlevel):
    if idx not in self.saved_comments:
        return
    key, comment, beginline = self.saved_comments[idx]
    currentlevel[key] = CommentValue(currentlevel[key], comment, beginline, self._dict)