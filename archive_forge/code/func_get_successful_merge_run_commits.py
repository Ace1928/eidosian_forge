from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
def get_successful_merge_run_commits(self) -> set[str]:
    """Return a set of recent successsful merge commits from Azure Pipelines."""
    parameters = dict(maxBuildsPerDefinition=100, queryOrder='queueTimeDescending', resultFilter='succeeded', reasonFilter='batchedCI', repositoryType=self.repo_type, repositoryId='%s/%s' % (self.org, self.project))
    url = '%s%s/_apis/build/builds?api-version=6.0&%s' % (self.org_uri, self.project, urllib.parse.urlencode(parameters))
    http = HttpClient(self.args, always=True)
    response = http.get(url)
    try:
        result = response.json()
    except Exception:
        display.warning('Unable to find project. Cannot determine changes. All tests will be executed.')
        return set()
    commits = set((build['sourceVersion'] for build in result['value']))
    return commits