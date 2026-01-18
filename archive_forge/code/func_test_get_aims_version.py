from ase.calculators.aims import get_aims_version
def test_get_aims_version():
    assert get_aims_version(version_string) == '200112.2'