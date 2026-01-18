import abc
from typing import List
import stevedore
def high_level_synthesis_plugin_names(op_name: str) -> List[str]:
    """Return a list of plugin names installed for a given high level object name

    Args:
        op_name: The operation name to find the installed plugins for. For example,
            if you provide ``"clifford"`` as the input it will find all the installed
            clifford synthesis plugins that can synthesize :class:`.Clifford` objects.
            The name refers to the :attr:`.Operation.name` attribute of the relevant objects.

    Returns:
        A list of installed plugin names for the specified high level operation

    """
    plugin_manager = HighLevelSynthesisPluginManager()
    return plugin_manager.method_names(op_name)