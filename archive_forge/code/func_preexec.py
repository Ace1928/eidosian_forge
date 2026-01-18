from __future__ import unicode_literals
import contextlib
import difflib
import io
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
def preexec():
    os.close(stdinpipe[1])
    os.close(stdoutpipe[0])