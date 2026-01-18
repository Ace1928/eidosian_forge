from typing import Iterable
import cirq
from cirq_web import widget
from cirq_web.circuits.symbols import (
def get_widget_bundle_name(self) -> str:
    return 'circuit.bundle.js'