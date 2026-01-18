import cirq
import cirq_web
import pytest
@pytest.mark.parametrize('qubit', [cirq.GridQubit(0, 0), cirq.LineQubit(0)])
def test_circuit_client_code(qubit):
    moment = cirq.Moment(cirq.H(qubit))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))
    circuit_obj = [{'wire_symbols': ['H'], 'location_info': [{'row': 0, 'col': 0}], 'color_info': ['yellow'], 'moment': 0}]
    moments = 1
    stripped_id = circuit.id.replace('-', '')
    expected_client_code = f'''\n        <button id="camera-reset">Reset Camera</button>\n        <button id="camera-toggle">Toggle Camera Type</button>\n        <script>\n        let viz_{stripped_id} = createGridCircuit({str(circuit_obj)}, {str(moments)}, "{circuit.id}", {circuit.padding_factor});\n\n        document.getElementById("camera-reset").addEventListener('click', ()  => {{\n        viz_{stripped_id}.scene.setCameraAndControls(viz_{stripped_id}.circuit);\n        }});\n\n        document.getElementById("camera-toggle").addEventListener('click', ()  => {{\n        viz_{stripped_id}.scene.toggleCamera(viz_{stripped_id}.circuit);\n        }});\n        </script>\n        '''
    assert strip_ws(circuit.get_client_code()) == strip_ws(expected_client_code)