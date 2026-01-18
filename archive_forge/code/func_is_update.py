import dns.exception
def is_update(flags):
    """Is the opcode in flags UPDATE?

    *flags*, an ``int``, the DNS message flags.

    Returns a ``bool``.
    """
    return from_flags(flags) == UPDATE