import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def compare_json_patches(a, b):
    """Return 0 if a and b describe the same JSON patch."""
    return jsonpatch.JsonPatch.from_string(a) == jsonpatch.JsonPatch.from_string(b)