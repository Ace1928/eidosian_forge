from types import CodeType as code, FunctionType as function
def copycode(template, changes):
    if hasattr(code, 'replace'):
        return template.replace(**{'co_' + k: v for k, v in changes.items()})
    names = ['argcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'lnotab', 'freevars', 'cellvars']
    if hasattr(code, 'co_kwonlyargcount'):
        names.insert(1, 'kwonlyargcount')
    if hasattr(code, 'co_posonlyargcount'):
        names.insert(1, 'posonlyargcount')
    values = [changes.get(name, getattr(template, 'co_' + name)) for name in names]
    return code(*values)