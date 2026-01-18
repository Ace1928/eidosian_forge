import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def make_factory_fixture(name):

    @pytest.fixture(scope='session')
    def _factory(factories):
        factories.require(name)
        return factories[name]
    _factory.__name__ = '{}_factory'.format(name)
    return _factory