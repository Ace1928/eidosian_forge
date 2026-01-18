from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def set_config_send_strict(self, value):
    br = branch.Branch.open('local')
    br.get_config_stack().set('send_strict', value)