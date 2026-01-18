from pyudev.device import Device
def make_init(qobject, socket_notifier):
    """
    Generates an initializer to observer the given ``monitor``
    (a :class:`~pyudev.Monitor`):

    ``parent`` is the parent :class:`~PyQt{4,5}.QtCore.QObject` of this
    object.  It is passed unchanged to the inherited constructor of
    :class:`~PyQt{4,5}.QtCore.QObject`.
    """

    def __init__(self, monitor, parent=None):
        qobject.__init__(self, parent)
        self._setup_notifier(monitor, socket_notifier)
    return __init__