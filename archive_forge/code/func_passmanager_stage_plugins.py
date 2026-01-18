import abc
from typing import List, Optional, Dict
import stevedore
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig
def passmanager_stage_plugins(stage: str) -> Dict[str, PassManagerStagePlugin]:
    """Return a dict with, for each stage name, the class type of the plugin.

    This function is useful for getting more information about a plugin:

    .. code-block:: python

        from qiskit.transpiler.preset_passmanagers.plugin import passmanager_stage_plugins
        routing_plugins = passmanager_stage_plugins('routing')
        basic_plugin = routing_plugins['basic']
        help(basic_plugin)

    .. code-block:: text

        Help on BasicSwapPassManager in module ...preset_passmanagers.builtin_plugins object:

        class BasicSwapPassManager(...preset_passmanagers.plugin.PassManagerStagePlugin)
         |  Plugin class for routing stage with :class:`~.BasicSwap`
         |
         |  Method resolution order:
         |      BasicSwapPassManager
         |      ...preset_passmanagers.plugin.PassManagerStagePlugin
         |      abc.ABC
         |      builtins.object
         ...

    Args:
        stage: The stage name to get

    Returns:
        dict: the key is the name of the plugin and the value is the class type for each.

    Raises:
       TranspilerError: If an invalid stage name is specified.
    """
    plugin_mgr = PassManagerStagePluginManager()
    try:
        manager = getattr(plugin_mgr, f'{stage}_plugins')
    except AttributeError as exc:
        raise TranspilerError(f'Passmanager stage {stage} not found') from exc
    return {name: manager[name].obj for name in manager.names()}