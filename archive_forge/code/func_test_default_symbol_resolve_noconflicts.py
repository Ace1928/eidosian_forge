import pytest
import rpy2.robjects.packages_utils as p_u
@pytest.mark.parametrize('symbol_mapping,expected_conflict,expected_resolution', [({'foo_bar': ['foo.bar'], 'foo': ['foo']}, {}, {}), ({'foo_bar': ['foo.bar', 'foo_bar'], 'foo': ['foo']}, {}, {'foo_bar': ['foo_bar'], 'foo_bar_': ['foo.bar']}), ({'foo_bar': ['foo.bar', 'foo_bar', 'foo_bar_'], 'foo': ['foo']}, {'foo_bar': ['foo.bar', 'foo_bar', 'foo_bar_']}, {})])
def test_default_symbol_resolve_noconflicts(symbol_mapping, expected_conflict, expected_resolution):
    conflicts, resolved_mapping = p_u.default_symbol_resolve(symbol_mapping)
    assert conflicts == expected_conflict
    assert resolved_mapping == expected_resolution