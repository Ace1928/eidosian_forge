import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
def gitlab_resource_to_cls(gitlab_resource: str, namespace: ModuleType) -> Type[RESTObject]:
    classes = CaseInsensitiveDict(namespace.__dict__)
    lowercase_class = gitlab_resource.replace('-', '')
    class_type = classes[lowercase_class]
    if TYPE_CHECKING:
        assert isinstance(class_type, type)
        assert issubclass(class_type, RESTObject)
    return class_type