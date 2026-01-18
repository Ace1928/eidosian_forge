import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_cloneCoroutineDeprecation(self) -> None:
    """
        Cloning a tag containing a coroutine is unsafe. To avoid breaking
        programs that only flatten the clone or only flatten the original,
        we deprecate old behavior rather than making it an error immediately.
        """

    async def asyncFunc() -> NoReturn:
        raise NotImplementedError
    coro = asyncFunc()
    tag = proto('123', coro, '789')
    try:
        self.assertWarns(DeprecationWarning, 'Cloning a Tag which contains a coroutine is unsafe, since the coroutine can run only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', sys.modules[Tag.__module__].__file__, tag.clone)
    finally:
        coro.close()