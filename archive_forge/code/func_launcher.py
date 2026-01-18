import functools
from debugpy.common import json, log, messaging, util
@property
def launcher(self):
    return self.session.launcher