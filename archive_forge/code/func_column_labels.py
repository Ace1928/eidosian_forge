import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
@property
def column_labels(self):
    """
        A tuple containing the ``Tkinter.Label`` widgets used to
        display the label of each column.  If this multi-column
        listbox was created without labels, then this will be an empty
        tuple.  These widgets will all be augmented with a
        ``column_index`` attribute, which can be used to determine
        which column they correspond to.  This can be convenient,
        e.g., when defining callbacks for bound events.
        """
    return tuple(self._labels)