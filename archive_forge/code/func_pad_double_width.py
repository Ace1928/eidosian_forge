import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def pad_double_width(self, pad_char):
    """
        Pad all double-width characters in self by appending `pad_char` to each.
        For East Asian language support.
        """
    east_asian_width = unicodedata.east_asian_width
    for i in range(len(self.data)):
        line = self.data[i]
        if isinstance(line, str):
            new = []
            for char in line:
                new.append(char)
                if east_asian_width(char) in 'WF':
                    new.append(pad_char)
            self.data[i] = ''.join(new)