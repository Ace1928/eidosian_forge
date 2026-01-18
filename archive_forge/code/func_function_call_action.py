from pyparsing import *
from sys import stdin, argv, exit
def function_call_action(self, text, loc, fun):
    """Code executed after recognising the whole function call"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('FUN_CALL:', fun)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    if len(self.function_arguments) != self.symtab.get_attribute(self.function_call_index):
        raise SemanticException("Wrong number of arguments for function '%s'" % fun.name)
    self.function_arguments.reverse()
    self.codegen.function_call(self.function_call_index, self.function_arguments)
    self.codegen.restore_used_registers()
    return_type = self.symtab.get_type(self.function_call_index)
    self.function_call_index = self.function_call_stack.pop()
    self.function_arguments = self.function_arguments_stack.pop()
    register = self.codegen.take_register(return_type)
    self.codegen.move(self.codegen.take_function_register(return_type), register)
    return register