import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def display_dict(d: Dict[str, Any], padding: int) -> None:
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            print(f'{' ' * padding}{k.replace('_', '-')}:')
            new_padding = padding + 2
            self.display(v, verbose=True, padding=new_padding, obj=v)
            continue
        print(f'{' ' * padding}{k.replace('_', '-')}: {v}')