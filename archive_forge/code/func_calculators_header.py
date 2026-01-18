import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, check_output
import zlib
import pytest
import numpy as np
import ase
from ase.utils import workdir, seterr
from ase.test.factories import (CalculatorInputs,
from ase.dependencies import all_dependencies
def calculators_header(config):
    try:
        factories = get_factories(config)
    except NoSuchCalculator as err:
        pytest.exit(f'No such calculator: {err}')
    configpaths = factories.executable_config_paths
    module = factories.datafiles_module
    yield ''
    yield 'Calculators'
    yield '==========='
    if not configpaths:
        configtext = 'No configuration file specified'
    else:
        configtext = ', '.join((str(path) for path in configpaths))
    yield f'Config: {configtext}'
    if module is None:
        datafiles_text = 'ase-datafiles package not installed'
    else:
        datafiles_text = str(Path(module.__file__).parent)
    yield f'Datafiles: {datafiles_text}'
    yield ''
    for name in sorted(factory_classes):
        if name in factories.builtin_calculators:
            continue
        factory = factories.factories.get(name)
        if factory is None:
            configinfo = 'not installed'
        elif hasattr(factory, 'importname'):
            import importlib
            module = importlib.import_module(factory.importname)
            configinfo = str(module.__path__[0])
        else:
            configtokens = []
            for varname, variable in vars(factory).items():
                configtokens.append(f'{varname}={variable}')
            configinfo = ', '.join(configtokens)
        enabled = factories.enabled(name)
        if enabled:
            version = '<unknown version>'
            if hasattr(factory, 'version'):
                try:
                    version = factory.version()
                except Exception:
                    pass
            name = f'{name}-{version}'
        run = '[x]' if enabled else '[ ]'
        line = f'  {run} {name:16} {configinfo}'
        yield line
    yield ''
    yield helpful_message
    yield ''
    for name in factories.requested_calculators:
        if not factories.is_adhoc(name) and (not factories.installed(name)):
            pytest.exit(f'Calculator "{name}" is not installed.  Please run "ase test --help-calculators" on how to install calculators')