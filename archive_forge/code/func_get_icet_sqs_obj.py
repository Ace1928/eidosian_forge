from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def get_icet_sqs_obj(self, material: Atoms | Structure, cluster_space: _ClusterSpace | None=None) -> float:
    """
        Get the SQS objective function.

        Args:
            material (ase Atoms or pymatgen Structure) : structure to
                compute SQS objective function.
        Kwargs:
            cluster_space (ClusterSpace) : ClusterSpace of the SQS search.

        Returns:
            float : the SQS objective function
        """
    if isinstance(material, Structure):
        material = AseAtomsAdaptor.get_atoms(material)
    cluster_space = cluster_space or self._get_cluster_space()
    return compare_cluster_vectors(cv_1=cluster_space.get_cluster_vector(material), cv_2=self.sqs_vector, orbit_data=cluster_space.orbit_data, **self._sqs_obj_kwargs)