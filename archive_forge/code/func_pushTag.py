from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def pushTag(self, tag):
    """Internal method called by handle_starttag when a tag is opened."""
    if self.currentTag is not None:
        self.currentTag.contents.append(tag)
    self.tagStack.append(tag)
    self.currentTag = self.tagStack[-1]
    if tag.name != self.ROOT_TAG_NAME:
        self.open_tag_counter[tag.name] += 1
    if tag.name in self.builder.preserve_whitespace_tags:
        self.preserve_whitespace_tag_stack.append(tag)
    if tag.name in self.builder.string_containers:
        self.string_container_stack.append(tag)