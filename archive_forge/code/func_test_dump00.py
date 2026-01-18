import pytest  # NOQA
def test_dump00(self):
    import srsly.ruamel_yaml
    data = None
    s = srsly.ruamel_yaml.round_trip_dump(data)
    assert s == 'null\n...\n'
    d = srsly.ruamel_yaml.round_trip_load(s)
    assert d == data