import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def add_result(self, analysis_result):
    self._results.append(analysis_result)