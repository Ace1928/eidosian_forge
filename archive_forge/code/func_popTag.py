from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def popTag(self):
    """Internal method called by _popToTag when a tag is closed."""
    tag = self.tagStack.pop()
    if tag.name in self.open_tag_counter:
        self.open_tag_counter[tag.name] -= 1
    if self.preserve_whitespace_tag_stack and tag == self.preserve_whitespace_tag_stack[-1]:
        self.preserve_whitespace_tag_stack.pop()
    if self.string_container_stack and tag == self.string_container_stack[-1]:
        self.string_container_stack.pop()
    if self.tagStack:
        self.currentTag = self.tagStack[-1]
    return self.currentTag