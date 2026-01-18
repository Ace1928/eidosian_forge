from wsme.exc import (ClientSideError, InvalidInput, MissingArgument,
def test_missingargument():
    e = MissingArgument('argname', 'error message')
    assert e.faultstring == 'Missing argument: "argname": error message', e.faultstring