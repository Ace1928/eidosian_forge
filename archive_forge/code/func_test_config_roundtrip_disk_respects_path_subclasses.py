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
def test_config_roundtrip_disk_respects_path_subclasses(pathy_fixture):
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_path = pathy_fixture / 'config.cfg'
    cfg.to_disk(cfg_path)
    new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()