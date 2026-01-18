import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
@fixture(scope='module')
def protocol_fake(request):
    uuid4_patcher = patch('uuid.uuid4')
    uuid4_mock = uuid4_patcher.start()
    uuid4_mock.return_value = uuid.UUID('11111111-1111-1111-1111-111111111111')
    from winrm.protocol import Protocol
    protocol_fake = Protocol(endpoint='http://windows-host:5985/wsman', transport='plaintext', username='john.smith', password='secret')
    protocol_fake.transport = TransportStub()

    def uuid4_patch_stop():
        uuid4_patcher.stop()
    request.addfinalizer(uuid4_patch_stop)
    return protocol_fake