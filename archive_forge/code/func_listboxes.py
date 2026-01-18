import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
@property
def listboxes(self):
    """
        A tuple containing the ``Tkinter.Listbox`` widgets used to
        display individual columns.  These widgets will all be
        augmented with a ``column_index`` attribute, which can be used
        to determine which column they correspond to.  This can be
        convenient, e.g., when defining callbacks for bound events.
        """
    return tuple(self._listboxes)