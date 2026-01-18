import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def files_one():
    yield commands.FileModifyCommand(b'foo\x83', kind_to_mode('file', False), None, b'content A\n')