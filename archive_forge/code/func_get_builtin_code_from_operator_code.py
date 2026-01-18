from tensorflow.python.util import all_util
def get_builtin_code_from_operator_code(opcode):
    """Return the builtin code of the given operator code.

  The following method is introduced to resolve op builtin code shortage
  problem. The new builtin operator will be assigned to the extended builtin
  code field in the flatbuffer schema. Those methods helps to hide builtin code
  details.

  Args:
    opcode: Operator code.

  Returns:
    The builtin code of the given operator code.
  """
    if hasattr(opcode, 'BuiltinCode') and callable(opcode.BuiltinCode):
        return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
    return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)