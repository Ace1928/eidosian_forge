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
def test_config_roundtrip_disk():
    cfg = Config().from_str(OPTIMIZER_CFG)
    with make_tempdir() as path:
        cfg_path = path / 'config.cfg'
        cfg.to_disk(cfg_path)
        new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()