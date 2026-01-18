from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def batch_write_vasp_input(transformed_structures: Sequence[TransformedStructure], vasp_input_set: type[VaspInputSet]=MPRelaxSet, output_dir: str='.', create_directory: bool=True, subfolder: Callable[[TransformedStructure], str] | None=None, include_cif: bool=False, **kwargs):
    """Batch write vasp input for a sequence of transformed structures to
    output_dir, following the format output_dir/{group}/{formula}_{number}.

    Args:
        transformed_structures: Sequence of TransformedStructures.
        vasp_input_set: pymatgen.io.vasp.sets.VaspInputSet to creates
            vasp input files from structures.
        output_dir: Directory to output files
        create_directory (bool): Create the directory if not present.
            Defaults to True.
        subfolder: Function to create subdirectory name from
            transformed_structure.
            e.g., lambda x: x.other_parameters["tags"][0] to use the first
            tag.
        include_cif (bool): Boolean indication whether to output a CIF as
            well. CIF files are generally better supported in visualization
            programs.
        **kwargs: Any kwargs supported by vasp_input_set.
    """
    for idx, struct in enumerate(transformed_structures):
        formula = re.sub('\\s+', '', struct.final_structure.formula)
        if subfolder is not None:
            subdir = subfolder(struct)
            dirname = f'{output_dir}/{subdir}/{formula}_{idx}'
        else:
            dirname = f'{output_dir}/{formula}_{idx}'
        struct.write_vasp_input(vasp_input_set, dirname, create_directory=create_directory, **kwargs)
        if include_cif:
            from pymatgen.io.cif import CifWriter
            writer = CifWriter(struct.final_structure)
            writer.write_file(os.path.join(dirname, f'{formula}.cif'))