from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def simulator_tracking(cls: type) -> type:
    """Modifies all methods to add default simulator style tracking.

    Args:
        cls (type): a subclass of :class:`pennylane.devices.Device`

    Returns
        type: The inputted class that has now been modified to update the tracker upon function calls.

    Simulator style tracking updates:

    * ``executions``: the number of unique circuits that would be required on quantum hardware
    * ``shots``: the number of shots
    * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
    * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions,
      such as for non-commuting measurements and batched parameters.
    * ``batches``: The number of times :meth:`~pennylane.devices.Device.execute` is called.
    * ``results``: The results of each call of :meth:`~pennylane.devices.Device.execute`
    * ``derivative_batches``: How many times :meth:`~pennylane.devices.Device.compute_derivatives` is called.
    * ``execute_and_derivative_batches``: How many times :meth:`~pennylane.devices.Device.execute_and_compute_derivatives`
      is called
    * ``vjp_batches``: How many times :meth:`~pennylane.devices.Device.compute_vjp` is called
    * ``execute_and_vjp_batches``: How many times :meth:`~pennylane.devices.Device.execute_and_compute_vjp` is called
    * ``jvp_batches``: How many times :meth:`~pennylane.devices.Device.compute_jvp` is called
    * ``execute_and_jvp_batches``: How many times :meth:`~pennylane.devices.Device.execute_and_compute_jvp` is called
    * ``derivatives``: How many circuits are submitted to :meth:`~pennylane.devices.Device.compute_derivatives`
      or :meth:`~pennylane.devices.Device.execute_and_compute_derivatives`.
    * ``vjps``: How many circuits are submitted to :meth:`pennylane.devices.Device.compute_vjp`
      or :meth:`~pennylane.devices.Device.execute_and_compute_vjp`
    * ``jvps``: How many circuits are submitted to :meth:`~pennylane.devices.Device.compute_jvp`
      or :meth:`~pennylane.devices.Device.execute_and_compute_jvp`


    .. code-block:: python

        @simulator_tracking
        @single_tape_support
        class MyDevice(qml.devices.Device):

            def execute(self, circuits, execution_config = qml.devices.DefaultExecutionConfig):
                return tuple(0.0 for c in circuits)

    >>> dev = MyDevice()
    >>> ops = [qml.S(0)]
    >>> measurements = [qml.expval(qml.X(0)), qml.expval(qml.Z(0))]
    >>> t = qml.tape.QuantumScript(ops, measurements,shots=50)
    >>> with dev.tracker:
    ...     dev.execute((t, ) )
    >>> dev.tracker.history
    {'batches': [1],
    'simulations': [1],
    'executions': [2],
    'results': [0.0],
    'shots': [100],
    'resources': [Resources(num_wires=1, num_gates=1, gate_types=defaultdict(<class 'int'>, {'S': 1}),
    gate_sizes=defaultdict(<class 'int'>, {1: 1}), depth=1, shots=Shots(total_shots=50,
    shot_vector=(ShotCopies(50 shots x 1),)))]}

    """
    if not issubclass(cls, Device):
        raise ValueError('simulator_tracking only accepts subclasses of pennylane.devices.Device')
    if hasattr(cls, '_applied_modifiers'):
        cls._applied_modifiers.append(simulator_tracking)
    else:
        cls._applied_modifiers = [simulator_tracking]
    cls.execute = _track_execute(cls.execute)
    modifier_map = {'compute_derivatives': _track_compute_derivatives, 'execute_and_compute_derivatives': _track_execute_and_compute_derivatives, 'compute_jvp': _track_compute_jvp, 'execute_and_compute_jvp': _track_execute_and_compute_jvp, 'compute_vjp': _track_compute_vjp, 'execute_and_compute_vjp': _track_execute_and_compute_vjp}
    for name, modifier in modifier_map.items():
        if getattr(cls, name) != getattr(Device, name):
            original = getattr(cls, name)
            setattr(cls, name, modifier(original))
    return cls