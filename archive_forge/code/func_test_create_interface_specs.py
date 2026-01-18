import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
def test_create_interface_specs():
    new_interface = create_interface_specs('MyInterface')
    assert new_interface.__base__ == TraitedSpec
    assert isinstance(new_interface(), TraitedSpec)
    assert new_interface.__name__ == 'MyInterface'
    assert not new_interface().get()
    new_interface = create_interface_specs('MyInterface', BaseClass=BaseInterfaceInputSpec)
    assert new_interface.__base__ == BaseInterfaceInputSpec
    assert isinstance(new_interface(), BaseInterfaceInputSpec)
    assert new_interface.__name__ == 'MyInterface'
    assert not new_interface().get()
    params = [('params1', 'string', ['my description']), ('params2_files', 'string', ['my description @']), ('params3', 'int, optional', ['useful option']), ('out_params', 'string', ['my out description'])]
    new_interface = create_interface_specs('MyInterface', params=params, BaseClass=BaseInterfaceInputSpec)
    assert new_interface.__base__ == BaseInterfaceInputSpec
    assert isinstance(new_interface(), BaseInterfaceInputSpec)
    assert new_interface.__name__ == 'MyInterface'
    current_params = new_interface().get()
    assert len(current_params) == 4
    assert 'params1' in current_params.keys()
    assert 'params2_files' in current_params.keys()
    assert 'params3' in current_params.keys()
    assert 'out_params' in current_params.keys()