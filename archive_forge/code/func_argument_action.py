from pyparsing import *
from sys import stdin, argv, exit
def argument_action(self, text, loc, arg):
    """Code executed after recognising each of function's arguments"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('ARGUMENT:', arg.exp)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    arg_ordinal = len(self.function_arguments)
    if not self.symtab.same_type_as_argument(arg.exp, self.function_call_index, arg_ordinal):
        raise SemanticException("Incompatible type for argument %d in '%s'" % (arg_ordinal + 1, self.symtab.get_name(self.function_call_index)))
    self.function_arguments.append(arg.exp)