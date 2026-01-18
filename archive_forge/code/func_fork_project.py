import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
def fork_project(self, project_name, timeout=50, interval=5, owner=None):
    path = 'projects/%s/fork' % urlutils.quote(str(project_name), '')
    fields = {}
    if owner is not None:
        fields['namespace'] = owner
    response = self._api_request('POST', path, fields=fields)
    if response.status == 404:
        raise ForkingDisabled(project_name)
    if response.status == 409:
        resp = json.loads(response.data)
        raise GitLabConflict(resp.get('message'))
    if response.status not in (200, 201):
        _unexpected_status(path, response)
    project = json.loads(response.data)
    deadline = time.time() + timeout
    while project['import_status'] not in ('finished', 'none'):
        mutter('import status is %s', project['import_status'])
        if time.time() > deadline:
            raise ProjectCreationTimeout(project['path_with_namespace'], timeout)
        time.sleep(interval)
        project = self._get_project(project['path_with_namespace'])
    return project