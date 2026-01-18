import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def get_top_graph_type(self):
    parent = self
    while True:
        parent_ = parent.get_parent_graph()
        if parent_ == parent:
            break
        parent = parent_
    return parent.obj_dict['type']