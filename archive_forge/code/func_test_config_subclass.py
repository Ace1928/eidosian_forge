import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
def test_config_subclass():
    with pytest.raises(TypeError, match='missing 2 required keyword-only'):
        ExampleConfig()
    with pytest.raises(ValueError, match='x must be positive'):
        ExampleConfig(x=0, y='foo')
    with pytest.raises(TypeError, match='unexpected keyword'):
        ExampleConfig(x=1, y='foo', z='bad config')
    with pytest.raises(TypeError, match='unexpected keyword'):
        EmptyConfig(z='bad config')
    cfg = ExampleConfig(x=1, y='foo')
    assert cfg.x == 1
    assert cfg['x'] == 1
    assert cfg['y'] == 'foo'
    assert cfg.y == 'foo'
    assert 'x' in cfg
    assert 'y' in cfg
    assert 'z' not in cfg
    assert len(cfg) == 2
    assert set(iter(cfg)) == {'x', 'y'}
    assert set(cfg.keys()) == {'x', 'y'}
    assert set(cfg.values()) == {1, 'foo'}
    assert set(cfg.items()) == {('x', 1), ('y', 'foo')}
    assert dir(cfg) == ['x', 'y']
    cfg.x = 2
    cfg['y'] = 'bar'
    assert cfg['x'] == 2
    assert cfg.y == 'bar'
    with pytest.raises(TypeError, match="can't be deleted"):
        del cfg.x
    with pytest.raises(TypeError, match="can't be deleted"):
        del cfg['y']
    assert cfg.x == 2
    assert cfg == cfg
    assert cfg == ExampleConfig(x=2, y='bar')
    assert cfg != ExampleConfig(x=3, y='baz')
    assert cfg != Config(x=2, y='bar')
    with pytest.raises(TypeError, match='y must be a str'):
        cfg['y'] = 5
    with pytest.raises(ValueError, match='x must be positive'):
        cfg.x = -5
    assert cfg.get('x', 10) == 2
    with pytest.raises(AttributeError):
        cfg.z = 5
    with pytest.raises(KeyError):
        cfg['z'] = 5
    with pytest.raises(AttributeError):
        cfg.z
    with pytest.raises(KeyError):
        cfg['z']
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg == cfg2
    assert cfg.__doc__ == 'Example configuration.'
    assert cfg2.__doc__ == 'Example configuration.'