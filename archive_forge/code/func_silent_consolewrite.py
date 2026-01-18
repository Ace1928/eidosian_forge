import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
@pytest.fixture(scope='module')
def silent_consolewrite():
    module = rinterface.callbacks
    name = 'consolewrite_print'
    backup_func = getattr(module, name)
    setattr(module, name, _noconsole)
    try:
        yield
    finally:
        setattr(module, name, backup_func)