from pytest import raises as assert_raises
from scipy._lib._pep440 import Version, parse
def test_legacy_version():
    assert parse('invalid') < Version('0.0.0')
    assert parse('1.9.0-f16acvda') < Version('1.0.0')