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
def test_make_config_positional_args_complex():

    @my_registry.cats('catsie.v890')
    def catsie_890(*args: Optional[Union[StrictBool, PositiveInt]]):
        assert args[0] == 123
        return args[0]
    cfg = {'config': {'@cats': 'catsie.v890', '*': [123, True, 1, False]}}
    assert my_registry.resolve(cfg)['config'] == 123
    cfg = {'config': {'@cats': 'catsie.v890', '*': [123, 'True']}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(cfg)