from dbus._compat import is_py3
def get_dbus_message(self):
    if len(self.args) > 1:
        s = str(self.args)
    else:
        s = ''.join(self.args)
    if isinstance(s, bytes):
        return s.decode('utf-8', 'replace')
    return s