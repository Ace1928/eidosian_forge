import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def display_list(self, data: List[Union[str, gitlab.base.RESTObject]], fields: List[str], **kwargs: Any) -> None:
    verbose = kwargs.get('verbose', False)
    for obj in data:
        if isinstance(obj, gitlab.base.RESTObject):
            self.display(get_dict(obj, fields), verbose=verbose, obj=obj)
        else:
            print(obj)
        print('')