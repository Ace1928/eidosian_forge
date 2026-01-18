import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def prompt_bool(self, question, allow_editor=False):
    """Prompt the user with a yes/no question.

        This may be overridden by self.auto.  It may also *set* self.auto.  It
        may also raise UserAbort.
        :param question: The question to ask the user.
        :return: True or False
        """
    if self.auto:
        return True
    alternatives_chars = 'yn'
    alternatives = '&yes\n&No'
    if allow_editor:
        alternatives_chars += 'e'
        alternatives += '\n&edit manually'
    alternatives_chars += 'fq'
    alternatives += '\n&finish\n&quit'
    choice = self.prompt(question, alternatives, 1)
    if choice is None:
        char = 'n'
    else:
        char = alternatives_chars[choice]
    if char == 'y':
        return True
    elif char == 'e' and allow_editor:
        raise UseEditor
    elif char == 'f':
        self.auto = True
        return True
    if char == 'q':
        raise errors.UserAbort()
    else:
        return False