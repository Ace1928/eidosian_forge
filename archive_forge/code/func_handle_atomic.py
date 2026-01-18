from __future__ import print_function
import re
import hashlib
def handle_atomic(self, lines):
    """We digest everything."""
    for line in lines:
        self.handle_line(line)