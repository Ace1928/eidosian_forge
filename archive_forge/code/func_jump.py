from pyparsing import *
from sys import stdin, argv, exit
def jump(self, relcode, opposite, label):
    """Generates a jump instruction
           relcode  - relational operator code
           opposite - generate normal or opposite jump
           label    - jump label
        """
    jump = self.OPPOSITE_JUMPS[relcode] if opposite else self.CONDITIONAL_JUMPS[relcode]
    self.newline_text('{0}\t{1}'.format(jump, label), True)