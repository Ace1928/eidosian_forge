from collections import namedtuple
from string import ascii_letters, digits
from _pydevd_bundle import pydevd_xml
import pydevconsole
import builtins as __builtin__  # Py3
def completions_to_xml(completions):
    valid_xml = pydevd_xml.make_valid_xml_value
    quote = pydevd_xml.quote
    msg = ['<xml>']
    for comp in completions:
        msg.append('<comp p0="')
        msg.append(valid_xml(quote(comp[0], '/>_= \t')))
        msg.append('" p1="')
        msg.append(valid_xml(quote(comp[1], '/>_= \t')))
        msg.append('" p2="')
        msg.append(valid_xml(quote(comp[2], '/>_= \t')))
        msg.append('" p3="')
        msg.append(valid_xml(quote(comp[3], '/>_= \t')))
        msg.append('"/>')
    msg.append('</xml>')
    return ''.join(msg)