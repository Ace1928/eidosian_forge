import docutils.parsers
import docutils.statemachine
from docutils.parsers.rst import states
from docutils import frontend, nodes, Component
from docutils.transforms import universal
def assert_has_content(self):
    """
        Throw an ERROR-level DirectiveError if the directive doesn't
        have contents.
        """
    if not self.content:
        raise self.error('Content block expected for the "%s" directive; none found.' % self.name)