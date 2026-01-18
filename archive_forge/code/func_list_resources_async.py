import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def list_resources_async(self, resource_type):
    assert resource_type in ['nodes', 'volumes']
    glob = globals()
    loc = locals()
    exec('\nimport asyncio\n@asyncio.coroutine\ndef _list_async(driver):\n    projects = [project.id for project in driver.projects]\n    loop = asyncio.get_event_loop()\n    futures = [\n        loop.run_in_executor(None, driver.ex_list_%s_for_project, p)\n        for p in projects\n    ]\n    retval = []\n    for future in futures:\n        result = yield from future\n        retval.extend(result)\n    return retval' % resource_type, glob, loc)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    return loop.run_until_complete(loc['_list_async'](loc['self']))