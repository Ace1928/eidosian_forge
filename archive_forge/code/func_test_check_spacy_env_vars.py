from pathlib import Path
from typing import Any, Dict
import pytest
import srsly
from typer.testing import CliRunner
from weasel import app
from weasel.cli.main import HELP
from weasel.util import get_git_version
def test_check_spacy_env_vars(project_dir: Path, monkeypatch: pytest.MonkeyPatch):
    project_dir / 'abc.txt'
    result = CliRunner().invoke(app, ['run', '--dry', 'create', str(project_dir)])
    assert result.exit_code == 0
    assert "You've set a `SPACY_CONFIG_OVERRIDES` environment variable" not in result.output
    assert "You've set a `SPACY_PROJECT_USE_GIT_VERSION` environment variable" not in result.output
    monkeypatch.setenv('SPACY_CONFIG_OVERRIDES', 'test')
    monkeypatch.setenv('SPACY_PROJECT_USE_GIT_VERSION', 'false')
    result = CliRunner().invoke(app, ['run', '--dry', 'create', str(project_dir)])
    assert result.exit_code == 0
    assert "You've set a `SPACY_CONFIG_OVERRIDES` environment variable" in result.output
    assert "You've set a `SPACY_PROJECT_USE_GIT_VERSION` environment variable" in result.output