import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def get_trace_class(self, trace_name):
    if trace_name not in self._class_map:
        trace_module = import_module('plotly.graph_objs')
        trace_class_name = self.class_strs_map[trace_name]
        self._class_map[trace_name] = getattr(trace_module, trace_class_name)
    return self._class_map[trace_name]