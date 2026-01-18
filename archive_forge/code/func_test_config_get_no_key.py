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
def test_config_get_no_key():
    runner = CliRunner()
    result = runner.invoke(dask.cli.config_get)
    assert result.exit_code == 2
    expected = "Usage: get [OPTIONS] KEY\nTry 'get --help' for help.\n\nError: Missing argument 'KEY'.\n"
    assert result.output == expected