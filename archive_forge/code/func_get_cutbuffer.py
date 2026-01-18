from kivy.core.clipboard import ClipboardBase
def get_cutbuffer(self):
    p = self._clip('out', 'primary')
    data, _ = p.communicate()
    return data.decode('utf8')