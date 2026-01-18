from . import _gi
def connect_detailed(self, callback, detail, *args, **kargs):
    """Same as GObject.Object.connect except there is no need to specify
            the signal name. In addition concats "::<detail>" to the signal name
            when connecting; for use with notifications like "notify" when a property
            changes.
            """
    return self.gobj.connect(self + '::' + detail, callback, *args, **kargs)