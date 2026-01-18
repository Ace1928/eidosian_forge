import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar
import numpy as np
import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor
from ._wires import wires_to_json
@classmethod
@lru_cache(1)
def consumes_types(cls) -> FrozenSet[Type[Operator]]:
    return frozenset((Tensor, qml.QubitCarry, qml.QubitSum, qml.Hamiltonian, qml.QubitUnitary, qml.DiagonalQubitUnitary, qml.Hadamard, qml.PauliX, qml.PauliY, qml.PauliZ, qml.X, qml.Y, qml.Z, qml.T, qml.S, qml.SX, qml.CNOT, qml.CH, qml.SWAP, qml.ECR, qml.SISWAP, qml.CSWAP, qml.CCZ, qml.Toffoli, qml.WireCut, qml.Hermitian, qml.Projector, qml.MultiRZ, qml.IsingXX, qml.IsingYY, qml.IsingZZ, qml.IsingXY, qml.PSWAP, qml.CPhaseShift00, qml.CPhaseShift01, qml.CPhaseShift10, qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.Rot, qml.U1, qml.U2, qml.U3, qml.SingleExcitation, qml.SingleExcitationMinus, qml.SingleExcitationPlus, qml.DoubleExcitation, qml.DoubleExcitationMinus, qml.DoubleExcitationPlus, qml.OrbitalRotation, qml.FermionicSWAP, qml.SpecialUnitary, qml.BasisState, qml.QubitStateVector, qml.StatePrep, qml.QubitDensityMatrix, qml.QutritUnitary, qml.TShift, qml.TClock, qml.TAdd, qml.TSWAP, qml.THermitian, qml.AmplitudeDamping, qml.GeneralizedAmplitudeDamping, qml.PhaseDamping, qml.DepolarizingChannel, qml.BitFlip, qml.ResetError, qml.PauliError, qml.PhaseFlip, qml.ThermalRelaxationError, qml.Rotation, qml.Squeezing, qml.Displacement, qml.Beamsplitter, qml.TwoModeSqueezing, qml.QuadraticPhase, qml.ControlledAddition, qml.ControlledPhase, qml.Kerr, qml.CrossKerr, qml.InterferometerUnitary, qml.CoherentState, qml.SqueezedState, qml.DisplacedSqueezedState, qml.ThermalState, qml.GaussianState, qml.FockState, qml.FockStateVector, qml.FockDensityMatrix, qml.CatState, qml.NumberOperator, qml.TensorN, qml.QuadX, qml.QuadP, qml.QuadOperator, qml.PolyXP, qml.FockStateProjector, qml.Identity, qml.ControlledQubitUnitary, qml.ControlledPhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.CRot, qml.CZ, qml.CY))