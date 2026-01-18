from glance.tests import functional
from glance.tests.utils import depends_on_exe
from glance.tests.utils import execute
from glance.tests.utils import skip_if_disabled
@depends_on_exe('sqlite3')
@skip_if_disabled
def test_big_int_mapping(self):
    """Ensure BigInteger not mapped to BIGINT"""
    self.cleanup()
    self.start_servers(**self.__dict__.copy())
    cmd = 'sqlite3 tests.sqlite ".schema"'
    exitcode, out, err = execute(cmd, raise_error=True)
    self.assertNotIn('BIGINT', out)
    self.stop_servers()