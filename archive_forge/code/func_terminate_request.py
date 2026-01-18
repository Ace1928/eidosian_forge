import os
import sys
import debugpy
from debugpy import launcher
from debugpy.common import json
from debugpy.launcher import debuggee
def terminate_request(request):
    del debuggee.wait_on_exit_predicates[:]
    request.respond({})
    debuggee.kill()