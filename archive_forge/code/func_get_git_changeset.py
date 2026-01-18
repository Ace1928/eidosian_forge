from __future__ import unicode_literals
import datetime
import os
import subprocess
def get_git_changeset():
    """Returns a numeric identifier of the latest git changeset.
    The result is the UTC timestamp of the changeset in YYYYMMDDHHMMSS format.
    This value isn't guaranteed to be unique, but collisions are very unlikely,
    so it's sufficient for generating the development version numbers.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        git_log = subprocess.Popen('git log --pretty=format:%ct --quiet -1 HEAD', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=repo_dir, universal_newlines=True)
        timestamp = git_log.communicate()[0]
        timestamp = datetime.datetime.utcfromtimestamp(int(timestamp))
    except Exception:
        return None
    return timestamp.strftime('%Y%m%d%H%M%S')