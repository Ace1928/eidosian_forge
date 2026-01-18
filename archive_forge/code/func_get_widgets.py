from kivy.properties import ObjectProperty, BooleanProperty
from kivy.uix.behaviors.button import ButtonBehavior
from weakref import ref
@staticmethod
def get_widgets(groupname):
    """Return a list of the widgets contained in a specific group. If the
        group doesn't exist, an empty list will be returned.

        .. note::

            Always release the result of this method! Holding a reference to
            any of these widgets can prevent them from being garbage collected.
            If in doubt, do::

                l = ToggleButtonBehavior.get_widgets('mygroup')
                # do your job
                del l

        .. warning::

            It's possible that some widgets that you have previously
            deleted are still in the list. The garbage collector might need
            to release other objects before flushing them.
        """
    groups = ToggleButtonBehavior.__groups
    if groupname not in groups:
        return []
    return [x() for x in groups[groupname] if x()][:]