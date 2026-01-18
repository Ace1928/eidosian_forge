from wsme.exc import (ClientSideError, InvalidInput, MissingArgument,
def test_clientside_error():
    e = ClientSideError('Test')
    assert e.faultstring == 'Test'