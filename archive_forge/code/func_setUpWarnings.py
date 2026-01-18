import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def setUpWarnings(self):
    super(_base_argon2_test, self).setUpWarnings()
    warnings.filterwarnings('ignore', '.*Using argon2pure backend.*')