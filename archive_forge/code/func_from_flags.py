import dns.exception
def from_flags(flags):
    """Extract an opcode from DNS message flags.

    *flags*, an ``int``, the DNS flags.

    Returns an ``int``.
    """
    return (flags & 30720) >> 11