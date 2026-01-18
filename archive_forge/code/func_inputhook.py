import gtk, gobject
def inputhook(context):
    """
    When the eventloop of prompt-toolkit is idle, call this inputhook.

    This will run the GTK main loop until the file descriptor
    `context.fileno()` becomes ready.

    :param context: An `InputHookContext` instance.
    """

    def _main_quit(*a, **kw):
        gtk.main_quit()
        return False
    gobject.io_add_watch(context.fileno(), gobject.IO_IN, _main_quit)
    gtk.main()