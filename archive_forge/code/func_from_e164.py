import dns.exception
import dns.name
import dns.resolver
from ._compat import string_types, maybe_decode
def from_e164(text, origin=public_enum_domain):
    """Convert an E.164 number in textual form into a Name object whose
    value is the ENUM domain name for that number.

    Non-digits in the text are ignored, i.e. "16505551212",
    "+1.650.555.1212" and "1 (650) 555-1212" are all the same.

    *text*, a ``text``, is an E.164 number in textual form.

    *origin*, a ``dns.name.Name``, the domain in which the number
    should be constructed.  The default is ``e164.arpa.``.

    Returns a ``dns.name.Name``.
    """
    parts = [d for d in text if d.isdigit()]
    parts.reverse()
    return dns.name.from_text('.'.join(parts), origin=origin)