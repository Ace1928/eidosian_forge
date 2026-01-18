import json
import pydoc
from kubernetes import client
def get_watch_argument_name(self, func):
    if PYDOC_FOLLOW_PARAM in pydoc.getdoc(func):
        return 'follow'
    else:
        return 'watch'