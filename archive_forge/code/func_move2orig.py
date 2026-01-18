from nipype.interfaces.ants import (
import os
import pytest
def move2orig():
    os.chdir(orig_dir)