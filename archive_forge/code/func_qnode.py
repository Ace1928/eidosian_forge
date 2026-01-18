import pennylane as qml
@qml.qnode(device, *kwargs, diff_method=diff_method, interface=interface)
def qnode(params, **circuit_kwargs):
    ansatz(params, wires=device.wires, **circuit_kwargs)
    return [getattr(qml, measure)(o) for o in observables]