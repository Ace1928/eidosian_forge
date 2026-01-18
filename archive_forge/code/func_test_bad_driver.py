import importlib.metadata as importlib_metadata
from stevedore import driver
from stevedore import exception
from stevedore import extension
from stevedore.tests import test_extension
from stevedore.tests import utils
def test_bad_driver(self):
    try:
        driver.DriverManager('stevedore.test.extension', 'e2')
    except ImportError:
        pass
    else:
        self.assertEqual(False, 'No error raised')