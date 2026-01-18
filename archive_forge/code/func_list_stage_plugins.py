import abc
from typing import List, Optional, Dict
import stevedore
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig
def list_stage_plugins(stage_name: str) -> List[str]:
    """Get a list of installed plugins for a stage.

    Args:
        stage_name: The stage name to get the plugin names for

    Returns:
        plugins: The list of installed plugin names for the specified stages

    Raises:
       TranspilerError: If an invalid stage name is specified.
    """
    plugin_mgr = PassManagerStagePluginManager()
    if stage_name == 'init':
        return plugin_mgr.init_plugins.names()
    elif stage_name == 'layout':
        return plugin_mgr.layout_plugins.names()
    elif stage_name == 'routing':
        return plugin_mgr.routing_plugins.names()
    elif stage_name == 'translation':
        return plugin_mgr.translation_plugins.names()
    elif stage_name == 'optimization':
        return plugin_mgr.optimization_plugins.names()
    elif stage_name == 'scheduling':
        return plugin_mgr.scheduling_plugins.names()
    else:
        raise TranspilerError(f'Invalid stage name: {stage_name}')