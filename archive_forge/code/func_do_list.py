import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_list(self) -> Union[gitlab.base.RESTObjectList, List[gitlab.base.RESTObject]]:
    if TYPE_CHECKING:
        assert isinstance(self.mgr, gitlab.mixins.ListMixin)
    try:
        result = self.mgr.list(**self.args)
    except Exception as e:
        cli.die('Impossible to list objects', e)
    return result