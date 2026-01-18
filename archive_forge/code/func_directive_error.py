import docutils.parsers
import docutils.statemachine
from docutils.parsers.rst import states
from docutils import frontend, nodes, Component
from docutils.transforms import universal
def directive_error(self, level, message):
    """
        Return a DirectiveError suitable for being thrown as an exception.

        Call "raise self.directive_error(level, message)" from within
        a directive implementation to return one single system message
        at level `level`, which automatically gets the directive block
        and the line number added.

        Preferably use the `debug`, `info`, `warning`, `error`, or `severe`
        wrapper methods, e.g. ``self.error(message)`` to generate an
        ERROR-level directive error.
        """
    return DirectiveError(level, message)