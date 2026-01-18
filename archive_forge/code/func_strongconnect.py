import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def strongconnect(node):
    index[node] = index_counter[0]
    lowlinks[node] = index_counter[0]
    index_counter[0] += 1
    stack.append(node)
    try:
        successors = graph[node]
    except Exception:
        successors = []
    for successor in successors:
        if successor not in lowlinks:
            strongconnect(successor)
            lowlinks[node] = min(lowlinks[node], lowlinks[successor])
        elif successor in stack:
            lowlinks[node] = min(lowlinks[node], index[successor])
    if lowlinks[node] == index[node]:
        connected_component = []
        while True:
            successor = stack.pop()
            connected_component.append(successor)
            if successor == node:
                break
        component = tuple(connected_component)
        result.append(component)