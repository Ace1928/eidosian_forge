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
def test_config_roundtrip_bytes():
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_bytes = cfg.to_bytes()
    new_cfg = Config().from_bytes(cfg_bytes)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()