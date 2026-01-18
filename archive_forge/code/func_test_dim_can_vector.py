from sympy.core.symbol import symbols
from sympy.matrices.dense import (Matrix, eye)
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.dimensions import DimensionSystem
def test_dim_can_vector():
    dimsys = DimensionSystem([length, mass, time], [velocity, action], {velocity: {length: 1, time: -1}})
    assert dimsys.dim_can_vector(length) == Matrix([1, 0, 0])
    assert dimsys.dim_can_vector(velocity) == Matrix([1, 0, -1])
    dimsys = DimensionSystem((length, velocity, action), (mass, time), {time: {length: 1, velocity: -1}})
    assert dimsys.dim_can_vector(length) == Matrix([0, 1, 0])
    assert dimsys.dim_can_vector(velocity) == Matrix([0, 0, 1])
    assert dimsys.dim_can_vector(time) == Matrix([0, 1, -1])
    dimsys = DimensionSystem((length, mass, time), (velocity, action), {velocity: {length: 1, time: -1}, action: {mass: 1, length: 2, time: -1}})
    assert dimsys.dim_vector(length) == Matrix([1, 0, 0])
    assert dimsys.dim_vector(velocity) == Matrix([1, 0, -1])