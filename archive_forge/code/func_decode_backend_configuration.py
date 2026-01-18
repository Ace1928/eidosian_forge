from typing import Dict, Union, List
import dateutil.parser
def decode_backend_configuration(config: Dict) -> None:
    """Decode backend configuration.

    Args:
        config: A ``QasmBackendConfiguration`` or ``PulseBackendConfiguration``
            in dictionary format.
    """
    config['online_date'] = dateutil.parser.isoparse(config['online_date'])
    if 'u_channel_lo' in config:
        for u_channle_list in config['u_channel_lo']:
            for u_channle_lo in u_channle_list:
                u_channle_lo['scale'] = _to_complex(u_channle_lo['scale'])