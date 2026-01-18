import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
def determine_encoding_from_data(self, data):
    """
        Try to determine the encoding of `data` by looking *in* `data`.
        Check for a byte order mark (BOM) or an encoding declaration.
        """
    for start_bytes, encoding in self.byte_order_marks:
        if data.startswith(start_bytes):
            return encoding
    for line in data.splitlines()[:2]:
        match = self.coding_slug.search(line)
        if match:
            return match.group(1).decode('ascii')
    return None