from .reference import Reference
from typing import Any, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike
@property
def tag(self) -> Union['TagObject', None]:
    """
        :return: Tag object this tag ref points to or None in case
            we are a lightweight tag"""
    obj = self.object
    if obj.type == 'tag':
        return obj
    return None