import sys, os
import textwrap
def enable_interspersed_args(self):
    """Set parsing to not stop on the first non-option, allowing
        interspersing switches with command arguments. This is the
        default behavior. See also disable_interspersed_args() and the
        class documentation description of the attribute
        allow_interspersed_args."""
    self.allow_interspersed_args = True