from . import win32
def set_console(self, attrs=None, on_stderr=False):
    if attrs is None:
        attrs = self.get_attrs()
    handle = win32.STDOUT
    if on_stderr:
        handle = win32.STDERR
    win32.SetConsoleTextAttribute(handle, attrs)