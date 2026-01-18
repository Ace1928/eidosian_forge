import sys, os
import textwrap
def disable_interspersed_args(self):
    """Set parsing to stop on the first non-option. Use this if
        you have a command processor which runs another command that
        has options of its own and you want to make sure these options
        don't get confused.
        """
    self.allow_interspersed_args = False