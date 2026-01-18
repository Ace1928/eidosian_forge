import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_list_bgp_sessions(self, ex_project_id=None):
    if ex_project_id:
        projects = [ex_project_id]
    elif self.project_id:
        projects = [self.project_id]
    else:
        projects = [p.id for p in self.projects]
    retval = []
    for p in projects:
        retval.extend(self.ex_list_bgp_sessions_for_project(p)['bgp_sessions'])
    return retval