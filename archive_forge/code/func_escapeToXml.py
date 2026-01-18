from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def escapeToXml(text, isattrib=0):
    """Escape text to proper XML form, per section 2.3 in the XML specification.

    @type text: C{str}
    @param text: Text to escape

    @type isattrib: C{bool}
    @param isattrib: Triggers escaping of characters necessary for use as
                     attribute values
    """
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    if isattrib == 1:
        text = text.replace("'", '&apos;')
        text = text.replace('"', '&quot;')
    return text