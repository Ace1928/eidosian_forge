import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
@property
def errors(self):
    return [log for log in self._log if log[0] == ERROR]