import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_blacklist(cls, client, blacklist_id):
    try:
        client.zone_blacklist_delete(blacklist_id)
    except CommandFailed:
        pass