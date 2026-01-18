from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def endData(self, containerClass=None):
    """Method called by the TreeBuilder when the end of a data segment
        occurs.
        """
    if self.current_data:
        current_data = ''.join(self.current_data)
        if not self.preserve_whitespace_tag_stack:
            strippable = True
            for i in current_data:
                if i not in self.ASCII_SPACES:
                    strippable = False
                    break
            if strippable:
                if '\n' in current_data:
                    current_data = '\n'
                else:
                    current_data = ' '
        self.current_data = []
        if self.parse_only and len(self.tagStack) <= 1 and (not self.parse_only.text or not self.parse_only.search(current_data)):
            return
        containerClass = self.string_container(containerClass)
        o = containerClass(current_data)
        self.object_was_parsed(o)