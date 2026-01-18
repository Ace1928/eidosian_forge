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
def add_stage(self, stage: dict | None=None, commands: str | list[str] | dict[str, str | float] | None=None, stage_name: str | None=None, after_stage: str | int | None=None) -> None:
    """
        Adds a new stage to the LammpsInputFile, either from a whole stage (dict) or
        from a stage_name and commands. Both ways are mutually exclusive.

        Examples:
            1) In order to add a stage defining the force field to be used, you can use:
            ```
            your_input_file.add_stage(
                commands=["pair_coeff 1 1 morse 0.0580 3.987 3.404", "pair_coeff 1 4 morse 0.0408 1.399 3.204"],
                stage_name="Definition of the force field"
            )
            ```
            or
            ```
            your_input_file.add_stage(
                {
                    "stage_name": "Definition of the force field",
                    "commands": [
                        ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                        ("pair_coeff", "1 4 morse 0.0408 1.399 3.204")
                    ],
                }
            )
            ```
            2) Another stage could consist in an energy minimization. In that case, the commands could look like
            ```
            commands = [
                "thermo 1",
                "thermo_style custom step lx ly lz press pxx pyy pzz pe",
                "dump dmp all atom 5 run.dump",
                "min_style cg",
                "fix 1 all box/relax iso 0.0 vmax 0.001",
                "minimize 1.0e-16 1.0e-16 5000 10000",
                "write_data run.data"
            ]
            ```
            or a dictionary such as `{"thermo": 1, ...}`, or a string with a single command (e.g., "units atomic").

        Args:
            stage (dict): if provided, this is the stage that will be added to the LammpsInputFile.stages
            commands (str or list or dict): if provided, these are the LAMMPS command(s)
                that will be included in the stage to add.
                Can pass a list of LAMMPS commands with their arguments.
                Also accepts a dictionary of LAMMPS commands and
                corresponding arguments as key, value pairs.
                A single string can also be passed (single command together with its arguments).
                Not used in case a whole stage is given.
            stage_name (str): If a stage name is mentioned, the commands are added
                under that stage block, else the new stage is named from numbering.
                If given, stage_name cannot be one of the already present stage names.
                Not used in case a whole stage is given.
            after_stage (str): Name of the stage after which this stage should be added.
                If None, the stage is added at the end of the LammpsInputFile.
        """
    if after_stage is None:
        index_insert = -1
    elif isinstance(after_stage, int):
        index_insert = after_stage + 1
    elif after_stage in self.stages_names:
        index_insert = self.stages_names.index(after_stage) + 1
        if index_insert == len(self.stages_names):
            index_insert = -1
    else:
        raise ValueError('The stage after which this one should be added does not exist.')
    if stage:
        if commands or stage_name:
            warnings.warn('A stage has been passed together with commands and stage_name. This is incompatible. Only the stage will be used.')
        if stage['stage_name'] in self.stages_names:
            raise ValueError('The provided stage name is already present in LammpsInputFile.stages.')
        self._check_stage_format(stage)
        if index_insert == -1:
            self.stages.append(stage)
        else:
            self.stages.insert(index_insert, stage)
    else:
        self._initialize_stage(stage_name=stage_name, stage_index=index_insert)
        if commands:
            self.add_commands(stage_name=self.stages_names[index_insert], commands=commands)