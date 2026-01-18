from typing import Dict, Union, List
import dateutil.parser
def decode_backend_properties(properties: Dict) -> None:
    """Decode backend properties.

    Args:
        properties: A ``BackendProperties`` in dictionary format.
    """
    properties['last_update_date'] = dateutil.parser.isoparse(properties['last_update_date'])
    for qubit in properties['qubits']:
        for nduv in qubit:
            nduv['date'] = dateutil.parser.isoparse(nduv['date'])
    for gate in properties['gates']:
        for param in gate['parameters']:
            param['date'] = dateutil.parser.isoparse(param['date'])
    for gen in properties['general']:
        gen['date'] = dateutil.parser.isoparse(gen['date'])