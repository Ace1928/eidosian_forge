from pexpect import ExceptionPexpect, TIMEOUT, EOF, spawn
import time
import os
import sys
import re
def set_unique_prompt(self):
    """This sets the remote prompt to something more unique than ``#`` or ``$``.
        This makes it easier for the :meth:`prompt` method to match the shell prompt
        unambiguously. This method is called automatically by the :meth:`login`
        method, but you may want to call it manually if you somehow reset the
        shell prompt. For example, if you 'su' to a different user then you
        will need to manually reset the prompt. This sends shell commands to
        the remote host to set the prompt, so this assumes the remote host is
        ready to receive commands.

        Alternatively, you may use your own prompt pattern. In this case you
        should call :meth:`login` with ``auto_prompt_reset=False``; then set the
        :attr:`PROMPT` attribute to a regular expression. After that, the
        :meth:`prompt` method will try to match your prompt pattern.
        """
    self.sendline('unset PROMPT_COMMAND')
    self.sendline(self.PROMPT_SET_SH)
    i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
    if i == 0:
        self.sendline(self.PROMPT_SET_CSH)
        i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
        if i == 0:
            self.sendline(self.PROMPT_SET_ZSH)
            i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
            if i == 0:
                return False
    return True