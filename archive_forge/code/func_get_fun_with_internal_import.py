import dill
def get_fun_with_internal_import():

    def fun_with_import():
        import re
        return re.compile('$')
    return fun_with_import