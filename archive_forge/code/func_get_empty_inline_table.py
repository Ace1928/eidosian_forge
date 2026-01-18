import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def get_empty_inline_table(self):

    class DynamicInlineTableDict(self._dict, InlineTableDict):
        """Concrete sentinel subclass for inline tables.
            It is a subclass of _dict which is passed in dynamically at load
            time

            It is also a subclass of InlineTableDict
            """
    return DynamicInlineTableDict()