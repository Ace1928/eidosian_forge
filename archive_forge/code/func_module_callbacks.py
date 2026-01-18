from __future__ import print_function
def module_callbacks():

    def is_in_coroutine_module(name):
        return name.startswith('coroutine.')

    def is_in_modules_module(name):
        if name in ['require', 'module'] or name.startswith('package'):
            return True
        else:
            return False

    def is_in_string_module(name):
        return name.startswith('string.')

    def is_in_table_module(name):
        return name.startswith('table.')

    def is_in_math_module(name):
        return name.startswith('math')

    def is_in_io_module(name):
        return name.startswith('io.')

    def is_in_os_module(name):
        return name.startswith('os.')

    def is_in_debug_module(name):
        return name.startswith('debug.')
    return {'coroutine': is_in_coroutine_module, 'modules': is_in_modules_module, 'string': is_in_string_module, 'table': is_in_table_module, 'math': is_in_math_module, 'io': is_in_io_module, 'os': is_in_os_module, 'debug': is_in_debug_module}