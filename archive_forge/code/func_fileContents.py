from typing import IO, Callable, Optional, TypeVar
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from twisted.python.filepath import IFilePath
from twisted.python.reflect import fullyQualifiedName
def fileContents(m: Matcher[str], encoding: str='utf-8') -> Matcher[IFilePath]:
    """
    Create a matcher which matches a L{FilePath} the contents of which are
    matched by L{m}.
    """

    def getContent(p: IFilePath) -> str:
        f: IO[bytes]
        with p.open() as f:
            return f.read().decode(encoding)
    return after(getContent, m)