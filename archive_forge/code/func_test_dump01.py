import pytest  # NOQA
def test_dump01(self):
    import srsly.ruamel_yaml
    data = None
    s = srsly.ruamel_yaml.round_trip_dump(data, explicit_end=True)
    assert s == 'null\n...\n'
    d = srsly.ruamel_yaml.round_trip_load(s)
    assert d == data