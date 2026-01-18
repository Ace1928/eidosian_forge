from __future__ import (absolute_import, division, print_function)
def check_sys_modules(path, before, messages):
    """Check for unwanted changes to sys.modules.
        :type path: str
        :type before: dict[str, module]
        :type messages: set[str]
        """
    after = sys.modules
    removed = set(before.keys()) - set(after.keys())
    changed = set((key for key, value in before.items() if key in after and value != after[key]))
    for module in sorted(removed):
        report_message(path, 0, 0, 'unload', 'unloading of "%s" in sys.modules is not supported' % module, messages)
    for module in sorted(changed):
        report_message(path, 0, 0, 'reload', 'reloading of "%s" in sys.modules is not supported' % module, messages)