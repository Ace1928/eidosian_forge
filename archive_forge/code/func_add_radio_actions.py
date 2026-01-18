import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def add_radio_actions(self, entries, value=None, on_change=None, user_data=None):
    """
            The add_radio_actions() method is a convenience method that creates a
            number of gtk.RadioAction objects based on the information in the list
            of action entry tuples contained in entries and adds them to the action
            group. The entry tuples can vary in size from one to six items with the
            following information:

                * The name of the action. Must be specified.
                * The stock id for the action. Optional with a default value of None
                  if a label is specified.
                * The label for the action. This field should typically be marked
                  for translation, see the set_translation_domain() method. Optional
                  with a default value of None if a stock id is specified.
                * The accelerator for the action, in the format understood by the
                  gtk.accelerator_parse() function. Optional with a default value of
                  None.
                * The tooltip for the action. This field should typically be marked
                  for translation, see the set_translation_domain() method. Optional
                  with a default value of None.
                * The value to set on the radio action. Optional with a default
                  value of 0. Should be specified in applications.

            The value parameter specifies the radio action that should be set
            active. The "changed" signal of the first radio action is connected to
            the on_change callback (if specified and not None) and the accel paths
            of the actions are set to <Actions>/group-name/action-name.
            """
    try:
        iter(entries)
    except TypeError:
        raise TypeError('entries must be iterable')
    first_action = None

    def _process_action(group_source, name, stock_id=None, label=None, accelerator=None, tooltip=None, entry_value=0):
        action = RadioAction(name=name, label=label, tooltip=tooltip, stock_id=stock_id, value=entry_value)
        if GTK3:
            action.join_group(group_source)
        if value == entry_value:
            action.set_active(True)
        self.add_action_with_accel(action, accelerator)
        return action
    for e in entries:
        action = _process_action(first_action, *e)
        if first_action is None:
            first_action = action
    if first_action is not None and on_change is not None:
        if user_data is None:
            first_action.connect('changed', on_change)
        else:
            first_action.connect('changed', on_change, user_data)