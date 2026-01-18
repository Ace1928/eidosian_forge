import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
@classmethod
def read_from_trajectory(cls, trajectory, frame=-1, atoms=None):
    """Read dynamics and atoms from trajectory (Class method).

        Simultaneously reads the atoms and the dynamics from a BundleTrajectory,
        including the internal data of the NPT dynamics object (automatically
        saved when attaching a BundleTrajectory to an NPT object).

        Arguments:

        trajectory
            The filename or an open BundleTrajectory object.

        frame (optional)
            Which frame to read.  Default: the last.

        atoms (optional, internal use only)
            Pre-read atoms.  Do not use.
        """
    if isinstance(trajectory, str):
        if trajectory.endswith('/'):
            trajectory = trajectory[:-1]
        if trajectory.endswith('.bundle'):
            from ase.io.bundletrajectory import BundleTrajectory
            trajectory = BundleTrajectory(trajectory)
        else:
            raise ValueError(f"Cannot open '{trajectory}': unsupported file format")
    if atoms is None:
        atoms = trajectory[frame]
    init_data = trajectory.read_extra_data('npt_init', 0)
    frame_data = trajectory.read_extra_data('npt_dynamics', frame)
    dyn = cls(atoms, timestep=init_data['dt'], temperature=init_data['temperature'], externalstress=init_data['externalstress'], ttime=init_data['ttime'], pfactor=init_data['pfactor_given'], mask=init_data['mask'])
    dyn.desiredEkin = init_data['desiredEkin']
    dyn.tfact = init_data['tfact']
    dyn.pfact = init_data['pfact']
    dyn.frac_traceless = init_data['frac_traceless']
    for k, v in frame_data.items():
        setattr(dyn, k, v)
    return (dyn, atoms)