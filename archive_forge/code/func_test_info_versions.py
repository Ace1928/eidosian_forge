from __future__ import annotations
import importlib
import json
import pathlib
import platform
import sys
import click
import pytest
import yaml
from click.testing import CliRunner
import dask
import dask.cli
from dask._compatibility import importlib_metadata
def test_info_versions():
    runner = CliRunner()
    result = runner.invoke(dask.cli.versions)
    assert result.exit_code == 0
    table = json.loads(result.output)
    assert table['Python'] == '.'.join((str(x) for x in sys.version_info[:3]))
    assert table['dask'] == dask.__version__
    assert table['Platform'] == platform.uname().system
    try:
        from distributed import __version__ as distributed_version
    except ImportError:
        distributed_version = None
    assert table['distributed'] == distributed_version