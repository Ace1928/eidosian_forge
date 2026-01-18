import os
import sys
from .. import bedding, osutils, tests
def override_whoami(test):
    test.overrideEnv('EMAIL', None)
    test.overrideEnv('BRZ_EMAIL', None)
    test.overrideAttr(bedding, '_auto_user_id', lambda: (None, None))