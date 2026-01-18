import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
def test_convert_to_traits_type():
    Params = namedtuple('Params', 'traits_type is_file')
    Res = namedtuple('Res', 'traits_type is_mandatory')
    l_entries = [Params('variable string', False), Params('variable int', False), Params('variable float', False), Params('variable bool', False), Params('variable complex', False), Params('variable int, optional', False), Params('variable string, optional', False), Params('variable float, optional', False), Params('variable bool, optional', False), Params('variable complex, optional', False), Params('string', False), Params('int', False), Params('string', True), Params('float', False), Params('bool', False), Params('complex', False), Params('string, optional', False), Params('int, optional', False), Params('string, optional', True), Params('float, optional', False), Params('bool, optional', False), Params('complex, optional', False)]
    l_expected = [Res(traits.ListStr, True), Res(traits.ListInt, True), Res(traits.ListFloat, True), Res(traits.ListBool, True), Res(traits.ListComplex, True), Res(traits.ListInt, False), Res(traits.ListStr, False), Res(traits.ListFloat, False), Res(traits.ListBool, False), Res(traits.ListComplex, False), Res(traits.Str, True), Res(traits.Int, True), Res(File, True), Res(traits.Float, True), Res(traits.Bool, True), Res(traits.Complex, True), Res(traits.Str, False), Res(traits.Int, False), Res(File, False), Res(traits.Float, False), Res(traits.Bool, False), Res(traits.Complex, False)]
    for entry, res in zip(l_entries, l_expected):
        traits_type, is_mandatory = convert_to_traits_type(entry.traits_type, entry.is_file)
        assert traits_type == res.traits_type
        assert is_mandatory == res.is_mandatory
    with pytest.raises(IOError):
        convert_to_traits_type('file, optional')