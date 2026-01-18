import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_commands(self, arg):
    """commands [bpnumber]
        (com) ...
        (com) end
        (Pdb)

        Specify a list of commands for breakpoint number bpnumber.
        The commands themselves are entered on the following lines.
        Type a line containing just 'end' to terminate the commands.
        The commands are executed when the breakpoint is hit.

        To remove all commands from a breakpoint, type commands and
        follow it immediately with end; that is, give no commands.

        With no bpnumber argument, commands refers to the last
        breakpoint set.

        You can use breakpoint commands to start your program up
        again.  Simply use the continue command, or step, or any other
        command that resumes execution.

        Specifying any command resuming execution (currently continue,
        step, next, return, jump, quit and their abbreviations)
        terminates the command list (as if that command was
        immediately followed by end).  This is because any time you
        resume execution (even with a simple next or step), you may
        encounter another breakpoint -- which could have its own
        command list, leading to ambiguities about which list to
        execute.

        If you use the 'silent' command in the command list, the usual
        message about stopping at a breakpoint is not printed.  This
        may be desirable for breakpoints that are to print a specific
        message and then continue.  If none of the other commands
        print anything, you will see no sign that the breakpoint was
        reached.
        """
    if not arg:
        bnum = len(bdb.Breakpoint.bpbynumber) - 1
    else:
        try:
            bnum = int(arg)
        except:
            self.error('Usage: commands [bnum]\n        ...\n        end')
            return
    try:
        self.get_bpbynumber(bnum)
    except ValueError as err:
        self.error('cannot set commands: %s' % err)
        return
    self.commands_bnum = bnum
    if bnum in self.commands:
        old_command_defs = (self.commands[bnum], self.commands_doprompt[bnum], self.commands_silent[bnum])
    else:
        old_command_defs = None
    self.commands[bnum] = []
    self.commands_doprompt[bnum] = True
    self.commands_silent[bnum] = False
    prompt_back = self.prompt
    self.prompt = '(com) '
    self.commands_defining = True
    try:
        self.cmdloop()
    except KeyboardInterrupt:
        if old_command_defs:
            self.commands[bnum] = old_command_defs[0]
            self.commands_doprompt[bnum] = old_command_defs[1]
            self.commands_silent[bnum] = old_command_defs[2]
        else:
            del self.commands[bnum]
            del self.commands_doprompt[bnum]
            del self.commands_silent[bnum]
        self.error('command definition aborted, old commands restored')
    finally:
        self.commands_defining = False
        self.prompt = prompt_back