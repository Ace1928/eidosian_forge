from pyparsing import *
from sys import stdin, argv, exit
def save_used_registers(self):
    """Pushes all used working registers before function call"""
    used = self.used_registers[:]
    del self.used_registers[:]
    self.used_registers_stack.append(used[:])
    used.sort()
    for reg in used:
        self.newline_text('PUSH\t%s' % SharedData.REGISTERS[reg], True)
    self.free_registers.extend(used)
    self.free_registers.sort(reverse=True)