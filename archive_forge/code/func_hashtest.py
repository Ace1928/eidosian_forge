import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def hashtest(version, t, logM, p, secret, salt, hex_digest, hash):
    return dict(version=version, rounds=t, logM=logM, memory_cost=1 << logM, parallelism=p, secret=secret, salt=salt, hex_digest=hex_digest, hash=hash)