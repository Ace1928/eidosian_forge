import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_zone_import(cls, client, zone_import_id):
    try:
        client.zone_import_delete(zone_import_id)
    except CommandFailed:
        pass