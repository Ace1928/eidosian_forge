import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
def test_config_fill_extra_fields():
    """Test that filling a config from a schema removes extra fields."""

    class TestSchemaContent(BaseModel):
        a: str
        b: int

        class Config:
            extra = 'forbid'

    class TestSchema(BaseModel):
        cfg: TestSchemaContent
    config = Config({'cfg': {'a': '1', 'b': 2, 'c': True}})
    with pytest.raises(ConfigValidationError):
        my_registry.fill(config, schema=TestSchema)
    filled = my_registry.fill(config, schema=TestSchema, validate=False)['cfg']
    assert filled == {'a': '1', 'b': 2}
    config2 = config.interpolate()
    filled = my_registry.fill(config2, schema=TestSchema, validate=False)['cfg']
    assert filled == {'a': '1', 'b': 2}
    config3 = Config({'cfg': {'a': '1', 'b': 2, 'c': True}}, is_interpolated=False)
    filled = my_registry.fill(config3, schema=TestSchema, validate=False)['cfg']
    assert filled == {'a': '1', 'b': 2}

    class TestSchemaContent2(BaseModel):
        a: str
        b: int

        class Config:
            extra = 'allow'

    class TestSchema2(BaseModel):
        cfg: TestSchemaContent2
    filled = my_registry.fill(config, schema=TestSchema2, validate=False)['cfg']
    assert filled == {'a': '1', 'b': 2, 'c': True}