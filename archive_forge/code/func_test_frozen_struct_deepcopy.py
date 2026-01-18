from typing import Dict, List
import catalogue
import pytest
from pytest import raises
from confection import Config, SimpleFrozenDict, SimpleFrozenList, registry
@pytest.mark.parametrize('frozen_type', ('dict', 'list'))
def test_frozen_struct_deepcopy(frozen_type):
    """Test whether setting default values for a FrozenDict/FrozenList works within a config, which utilizes
    deepcopy."""
    registry.bar = catalogue.create('confection', 'bar', entry_points=False)

    @registry.bar.register('foo_dict.v1')
    def make_dict(values: Dict[str, int]=SimpleFrozenDict(x=3)):
        return values

    @registry.bar.register('foo_list.v1')
    def make_list(values: List[int]=SimpleFrozenList([1, 2, 3])):
        return values
    cfg = Config()
    resolved = registry.resolve(cfg.from_str(f'\n            [something]\n            @bar = "foo_{frozen_type}.v1"        \n            '))
    assert isinstance(resolved['something'], Dict if frozen_type == 'dict' else List)