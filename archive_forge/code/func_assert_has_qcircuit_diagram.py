import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def assert_has_qcircuit_diagram(actual: cirq.Circuit, desired: str, **kwargs) -> None:
    """Determines if a given circuit has the desired qcircuit diagram.

    Args:
        actual: The circuit that was actually computed by some process.
        desired: The desired qcircuit diagram as a string. Newlines at the
            beginning and whitespace at the end are ignored.
        **kwargs: Keyword arguments to be passed to
            circuit_to_latex_using_qcircuit.
    """
    actual_diagram = ccq.circuit_to_latex_using_qcircuit(actual, **kwargs).lstrip('\n').rstrip()
    desired_diagram = desired.lstrip('\n').rstrip()
    assert actual_diagram == desired_diagram, f"Circuit's qcircuit diagram differs from the desired diagram.\n\nDiagram of actual circuit:\n{actual_diagram}\n\nDesired qcircuit diagram:\n{desired_diagram}\n\nHighlighted differences:\n{ct.highlight_text_differences(actual_diagram, desired_diagram)}\n"