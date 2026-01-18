import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def removeTest():
    del unittest.TestProgram.test