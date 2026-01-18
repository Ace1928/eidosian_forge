from typing import IO, Callable, Optional, TypeVar
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from twisted.python.filepath import IFilePath
from twisted.python.reflect import fullyQualifiedName
def getContent(p: IFilePath) -> str:
    f: IO[bytes]
    with p.open() as f:
        return f.read().decode(encoding)