from typing import Iterable
import cirq
from cirq_web import widget
from cirq_web.circuits.symbols import (
def get_client_code(self) -> str:
    stripped_id = self.id.replace('-', '')
    moments = len(self.circuit.moments)
    self.serialized_circuit = self._serialize_circuit()
    return f'''\n            <button id="camera-reset">Reset Camera</button>\n            <button id="camera-toggle">Toggle Camera Type</button>\n            <script>\n            let viz_{stripped_id} = createGridCircuit({self.serialized_circuit}, {moments}, "{self.id}", {self.padding_factor});\n\n            document.getElementById("camera-reset").addEventListener('click', ()  => {{\n            viz_{stripped_id}.scene.setCameraAndControls(viz_{stripped_id}.circuit);\n            }});\n\n            document.getElementById("camera-toggle").addEventListener('click', ()  => {{\n            viz_{stripped_id}.scene.toggleCamera(viz_{stripped_id}.circuit);\n            }});\n            </script>\n        '''