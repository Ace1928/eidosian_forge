from typing import Dict, Optional, Tuple, Type, Union
import dns.name
from dns.dnssecalgs.base import GenericPrivateKey
from dns.dnssectypes import Algorithm
from dns.exception import UnsupportedAlgorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
def get_algorithm_cls(algorithm: Union[int, str], prefix: AlgorithmPrefix=None) -> Type[GenericPrivateKey]:
    """Get Private Key class from Algorithm.

    *algorithm*, a ``str`` or ``int`` specifying the DNSKEY algorithm.

    Raises ``UnsupportedAlgorithm`` if the algorithm is unknown.

    Returns a ``dns.dnssecalgs.GenericPrivateKey``
    """
    algorithm = Algorithm.make(algorithm)
    cls = algorithms.get((algorithm, prefix))
    if cls:
        return cls
    raise UnsupportedAlgorithm('algorithm "%s" not supported by dnspython' % Algorithm.to_text(algorithm))