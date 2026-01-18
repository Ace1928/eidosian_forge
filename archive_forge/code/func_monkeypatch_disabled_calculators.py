import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def monkeypatch_disabled_calculators(self):
    test_calculator_names = self.autoenabled_calculators | self.builtin_calculators | self.requested_calculators
    disable_names = self.monkeypatch_calculator_constructors - test_calculator_names
    for name in disable_names:
        try:
            cls = get_calculator_class(name)
        except ImportError:
            pass
        else:

            def get_mock_init(name):

                def mock_init(obj, *args, **kwargs):
                    pytest.skip(f'use --calculators={name} to enable')
                return mock_init

            def mock_del(obj):
                pass
            cls.__init__ = get_mock_init(name)
            cls.__del__ = mock_del