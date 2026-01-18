import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_cloneGeneratorDeprecation(self) -> None:
    """
        Cloning a tag containing a generator is unsafe. To avoid breaking
        programs that only flatten the clone or only flatten the original,
        we deprecate old behavior rather than making it an error immediately.
        """
    tag = proto((str(n) for n in range(10)))
    self.assertWarns(DeprecationWarning, 'Cloning a Tag which contains a generator is unsafe, since the generator can be consumed only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', sys.modules[Tag.__module__].__file__, tag.clone)