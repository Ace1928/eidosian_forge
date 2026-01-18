import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_zone(cls, client, zone_id):
    try:
        client.zone_delete(zone_id)
    except CommandFailed:
        pass