import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
@pytest.fixture(scope='module')
def silent_console_print():
    with utils.obj_in_module(rpy2.rinterface_lib.callbacks, 'consolewrite_print', _just_pass):
        yield