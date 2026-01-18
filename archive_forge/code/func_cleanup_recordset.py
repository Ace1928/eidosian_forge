import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_recordset(cls, client, zone_id, recordset_id):
    try:
        client.recordset_delete(zone_id, recordset_id)
    except CommandFailed:
        pass