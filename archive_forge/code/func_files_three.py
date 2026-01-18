import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def files_three():
    yield commands.FileModifyCommand(b'foo', kind_to_mode('file', False), None, b'content A\n')