import os.path
import pytest
from tempfile import TemporaryDirectory
def test_logstart_inaccessible_file():
    with pytest.raises(IOError):
        _ip.logger.logstart(logfname='/')
    try:
        _ip.run_cell('a=1')
    finally:
        _ip.logger.log_active = False