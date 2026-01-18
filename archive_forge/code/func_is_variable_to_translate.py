def is_variable_to_translate(cls_name, var_name):
    if var_name in ('variablesReference', 'frameId', 'threadId'):
        return True
    if cls_name == 'StackFrame' and var_name == 'id':
        return True
    if cls_name == 'Thread' and var_name == 'id':
        return True
    return False