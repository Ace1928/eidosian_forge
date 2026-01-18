import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def get_dict(obj: Union[str, Dict[str, Any], gitlab.base.RESTObject], fields: List[str]) -> Union[str, Dict[str, Any]]:
    if not isinstance(obj, gitlab.base.RESTObject):
        return obj
    if fields:
        return {k: v for k, v in obj.attributes.items() if k in fields}
    return obj.attributes