import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_create(self) -> gitlab.base.RESTObject:
    if TYPE_CHECKING:
        assert isinstance(self.mgr, gitlab.mixins.CreateMixin)
    try:
        result = self.mgr.create(self.args)
    except Exception as e:
        cli.die('Impossible to create object', e)
    return result