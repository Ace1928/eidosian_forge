import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand
def dynamic_load_commands_in_register() -> List[Tuple[Union[Type[BaseOptimumCLICommand], CommandInfo], Optional[Type[BaseOptimumCLICommand]]]]:
    """
    Loads a list of command classes to register to the CLI from the `optimum/commands/register/` directory.
    It will look in any python file if there is a `REGISTER_COMMANDS` list, and load the commands to register
    accordingly.
    At this point, nothing is actually registered in the root CLI.

    Returns:
        `List[Tuple[Union[Type[BaseOptimumCLICommand], CommandInfo], Optional[Type[BaseOptimumCLICommand]]]]`: A list
        of tuples with two elements. The first element corresponds to the command to register, it can either be a
        subclass of `BaseOptimumCLICommand` or a `CommandInfo`. The second element corresponds to the parent command,
        where `None` means that the parent command is the root CLI command.
    """
    commands_to_register = []
    register_dir_path = Path(__file__).parent / 'register'
    for filename in register_dir_path.iterdir():
        if filename.is_dir() or filename.suffix != '.py':
            if filename.name not in ['__pycache__', 'README.md']:
                logger.warning(f'Skipping {filename} because only python files are allowed when registering commands dynamically.')
            continue
        module_name = f'.register.{filename.stem}'
        module = importlib.import_module(module_name, package='optimum.commands')
        commands_to_register_in_file = getattr(module, 'REGISTER_COMMANDS', [])
        for command_idx, command in enumerate(commands_to_register_in_file):
            if isinstance(command, tuple):
                command_or_command_info, parent_command_cls = command
            else:
                command_or_command_info = command
                parent_command_cls = None
            if not isinstance(command_or_command_info, CommandInfo) and (not issubclass(command_or_command_info, BaseOptimumCLICommand)):
                raise ValueError(f'The command at index {command_idx} in {filename} is not of the right type: {type(command_or_command_info)}.')
            commands_to_register.append((command_or_command_info, parent_command_cls))
    return commands_to_register