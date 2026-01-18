import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def parseString(self, instring, parseAll=False):
    """Execute the parse expression with the given string.
           This is the main interface to the client code, once the complete
           expression has been built.

           If you want the grammar to require that the entire input string be
           successfully parsed, then set C{parseAll} to True (equivalent to ending
           the grammar with C{L{StringEnd()}}).

           Note: C{parseString} implicitly calls C{expandtabs()} on the input string,
           in order to report proper column numbers in parse actions.
           If the input string contains tabs and
           the grammar uses parse actions that use the C{loc} argument to index into the
           string being parsed, you can ensure you have a consistent view of the input
           string by:
            - calling C{parseWithTabs} on your grammar before calling C{parseString}
              (see L{I{parseWithTabs}<parseWithTabs>})
            - define your parse action using the full C{(s,loc,toks)} signature, and
              reference the input string using the parse action's C{s} argument
            - explictly expand the tabs in your input string before calling
              C{parseString}
        """
    ParserElement.resetCache()
    if not self.streamlined:
        self.streamline()
    for e in self.ignoreExprs:
        e.streamline()
    if not self.keepTabs:
        instring = instring.expandtabs()
    try:
        loc, tokens = self._parse(instring, 0)
        if parseAll:
            loc = self.preParse(instring, loc)
            se = Empty() + StringEnd()
            se._parse(instring, loc)
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc
    else:
        return tokens