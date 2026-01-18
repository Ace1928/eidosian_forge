import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
def get_so_suffix():
    ret = sysconfig.get_config_var('EXT_SUFFIX')
    assert ret
    return ret