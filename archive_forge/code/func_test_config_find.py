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
def test_config_find(tmp_conf_dir):
    runner = CliRunner()
    result = runner.invoke(dask.cli.config_find, ['fizz.buzz'], catch_exceptions=False)
    expected = f'Unable to find [fizz.buzz] in any of the following paths:\n{tmp_conf_dir}\n'
    assert result.output == expected
    conf1 = tmp_conf_dir / 'conf1.yaml'
    conf2 = tmp_conf_dir / 'conf2.yaml'
    conf3 = tmp_conf_dir / 'conf3.yaml'
    conf1.write_text(yaml.dump({'fizz': {'buzz': 1}}))
    conf2.write_text(yaml.dump({'fizz': {'buzz': 2}}))
    conf3.write_text(yaml.dump({'foo': {'bar': 1}}))
    result = runner.invoke(dask.cli.config_find, ['fizz.buzz'], catch_exceptions=False)
    expected = f'Found [fizz.buzz] in the following files:\n{conf1}  [fizz.buzz=1]\n{conf2}  [fizz.buzz=2]\n'
    assert result.output == expected