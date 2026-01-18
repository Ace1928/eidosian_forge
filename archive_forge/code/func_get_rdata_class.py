from io import BytesIO
import base64
import binascii
import dns.exception
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.wiredata
from ._compat import xrange, string_types, text_type
def get_rdata_class(rdclass, rdtype):

    def import_module(name):
        with _import_lock:
            mod = __import__(name)
            components = name.split('.')
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod
    mod = _rdata_modules.get((rdclass, rdtype))
    rdclass_text = dns.rdataclass.to_text(rdclass)
    rdtype_text = dns.rdatatype.to_text(rdtype)
    rdtype_text = rdtype_text.replace('-', '_')
    if not mod:
        mod = _rdata_modules.get((dns.rdatatype.ANY, rdtype))
        if not mod:
            try:
                mod = import_module('.'.join([_module_prefix, rdclass_text, rdtype_text]))
                _rdata_modules[rdclass, rdtype] = mod
            except ImportError:
                try:
                    mod = import_module('.'.join([_module_prefix, 'ANY', rdtype_text]))
                    _rdata_modules[dns.rdataclass.ANY, rdtype] = mod
                except ImportError:
                    mod = None
    if mod:
        cls = getattr(mod, rdtype_text)
    else:
        cls = GenericRdata
    return cls