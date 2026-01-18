import sys
import pytest
from requests.help import info
def test_idna_with_version_attribute(mocker):
    """Verify we're actually setting idna version when it should be available."""
    mocker.patch('requests.help.idna', new=VersionedPackage('2.6'))
    assert info()['idna'] == {'version': '2.6'}