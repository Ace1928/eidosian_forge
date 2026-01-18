import pytest
import rpy2.robjects.packages_utils as p_u
def test___map_symbols():
    rnames = ('foo.bar', 'foo_bar', 'foo')
    translations = {}
    symbol_mapping, conflicts, resolutions = p_u._map_symbols(rnames, translations)
    expected_symbol_mapping = {'foo_bar': ['foo.bar', 'foo_bar'], 'foo': ['foo']}
    for new_symbol, old_symbols in expected_symbol_mapping.items():
        assert symbol_mapping[new_symbol] == old_symbols
    translations = {'foo.bar': 'foo_dot_bar'}
    symbol_mapping, conflicts, resolutions = p_u._map_symbols(rnames, translations)