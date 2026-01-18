from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
def remove_command(self, command: str, stage_name: str | list[str] | None=None, remove_empty_stages: bool=True) -> None:
    """
        Removes a given command from a given stage. If no stage is given, removes all occurrences of the command.
        In case removing a command completely empties a stage, the choice whether to keep this stage in the
        LammpsInputFile is given by remove_empty_stages.

        Args:
            command (str): command to be removed.
            stage_name (str or list): names of the stages where the command should be removed.
            remove_empty_stages (bool): whether to remove the stages emptied by removing the command or not.
        """
    if stage_name is None:
        stage_name = self.stages_names
    elif isinstance(stage_name, str):
        stage_name = [stage_name]
    elif not isinstance(stage_name, list):
        raise ValueError('If given, stage_name should be a string or a list of strings.')
    n_removed = 0
    indices_to_remove = []
    new_list_of_stages = []
    for i_stage, stage in enumerate(self.stages):
        if stage['stage_name'] in stage_name:
            new_commands = []
            for i_cmd, (cmd, arg) in enumerate(stage['commands']):
                if cmd == command:
                    n_removed += 1
                else:
                    new_commands.append((cmd, arg))
                    indices_to_remove.append([i_stage, i_cmd])
            if new_commands or not remove_empty_stages:
                new_list_of_stages.append({'stage_name': stage['stage_name'], 'commands': new_commands})
        else:
            new_list_of_stages.append(stage)
    self.stages = new_list_of_stages
    if n_removed == 0:
        warnings.warn(f'{command} not found in the LammpsInputFile.')