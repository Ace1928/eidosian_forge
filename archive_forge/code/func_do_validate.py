import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_validate(self) -> None:
    if TYPE_CHECKING:
        assert isinstance(self.mgr, gitlab.v4.objects.CiLintManager)
    try:
        self.mgr.validate(self.args)
    except GitlabCiLintError as e:
        cli.die('CI YAML Lint failed', e)
    except Exception as e:
        cli.die('Cannot validate CI YAML', e)