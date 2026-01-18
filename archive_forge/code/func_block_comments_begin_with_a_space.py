import ast
from hacking import core
import re
@core.flake8ext
def block_comments_begin_with_a_space(physical_line, line_number):
    """There should be a space after the # of block comments.

    There is already a check in pep8 that enforces this rule for
    inline comments.

    Okay: # this is a comment
    Okay: #!/usr/bin/python
    Okay: #  this is a comment
    K002: #this is a comment

    """
    MESSAGE = "K002 block comments should start with '# '"
    if line_number == 1 and physical_line.startswith('#!'):
        return
    text = physical_line.strip()
    if text.startswith('#'):
        if len(text) > 1 and (not text[1].isspace()):
            return (physical_line.index('#'), MESSAGE)