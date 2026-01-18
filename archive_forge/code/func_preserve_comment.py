import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def preserve_comment(self, line_no, key, comment, beginline):
    self.saved_comments[line_no] = (key, comment, beginline)