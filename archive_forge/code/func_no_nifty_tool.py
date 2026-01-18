import os
import pytest
from ....utils.filemanip import which
from ....testing import example_data
from .. import (
def no_nifty_tool(cmd=None):
    return which(cmd) is None