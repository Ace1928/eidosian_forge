import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def get_current_app():
    import traceback
    print('-- USES CURRENT_APP', file=sys.stderr)
    traceback.print_stack(file=sys.stderr)
    return _get_current_app()