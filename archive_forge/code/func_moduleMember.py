from pyside2uic.Compiler.indenter import write_code
def moduleMember(module, name):
    if module:
        return '%s.%s' % (module, name)
    return name