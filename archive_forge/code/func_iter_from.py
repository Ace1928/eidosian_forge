from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def iter_from(self, start):
    root = self.__root
    curr = start[1]
    while curr is not root:
        yield curr[2]
        curr = curr[1]