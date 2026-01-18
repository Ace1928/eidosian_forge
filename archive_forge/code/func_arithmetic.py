from pyparsing import *
from sys import stdin, argv, exit
def arithmetic(self, operation, operand1, operand2, operand3=None):
    """Generates an arithmetic instruction
           operation - one of supporetd operations
           operandX - index in symbol table or text representation of operand
           First two operands are input, third one is output
        """
    if isinstance(operand1, int):
        output_type = self.symtab.get_type(operand1)
        self.free_if_register(operand1)
    else:
        output_type = None
    if isinstance(operand2, int):
        output_type = self.symtab.get_type(operand2) if output_type == None else output_type
        self.free_if_register(operand2)
    else:
        output_type = SharedData.TYPES.NO_TYPE if output_type == None else output_type
    output = self.take_register(output_type) if operand3 == None else operand3
    mnemonic = self.arithmetic_mnemonic(operation, output_type)
    self.newline_text('{0}\t{1},{2},{3}'.format(mnemonic, self.symbol(operand1), self.symbol(operand2), self.symbol(output)), True)
    return output