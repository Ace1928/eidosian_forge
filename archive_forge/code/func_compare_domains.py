from typing import Any, Optional
def compare_domains(client_universe: str, credentials: Any) -> bool:
    """Returns True iff the universe domains used by the client and credentials match.

    Args:
        client_universe (str): The universe domain configured via the client options.
        credentials Any: The credentials being used in the client.

    Returns:
        bool: True iff client_universe matches the universe in credentials.

    Raises:
        ValueError: when client_universe does not match the universe in credentials.
    """
    credentials_universe = getattr(credentials, 'universe_domain', DEFAULT_UNIVERSE)
    if client_universe != credentials_universe:
        raise UniverseMismatchError(client_universe, credentials_universe)
    return True