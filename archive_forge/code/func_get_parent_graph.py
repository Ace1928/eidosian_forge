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
def get_parent_graph(self):
    return self.obj_dict.get('parent_graph', None)