from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def parse_binary(self, data, display, rawdict=0):
    """values, remdata = s.parse_binary(data, display, rawdict = 0)

        Convert a binary representation of the structure into Python values.

        DATA is a string or a buffer containing the binary data.
        DISPLAY should be a Xlib.protocol.display.Display object if
        there are any Resource fields or Lists with ResourceObjs.

        The Python values are returned as VALUES.  If RAWDICT is true,
        a Python dictionary is returned, where the keys are field
        names and the values are the corresponding Python value.  If
        RAWDICT is false, a DictWrapper will be returned where all
        fields are available as attributes.

        REMDATA are the remaining binary data, unused by the Struct object.

        """
    code = 'def parse_binary(self, data, display, rawdict = 0):\n  ret = {}\n  val = struct.unpack("%s", data[:%d])\n' % (self.static_codes, self.static_size)
    lengths = {}
    formats = {}
    vno = 0
    fno = 0
    for f in self.static_fields:
        if not f.name:
            pass
        elif isinstance(f, LengthField):
            if f.parse_value is None:
                lengths[f.name] = 'val[%d]' % vno
            else:
                lengths[f.name] = 'self.static_fields[%d].parse_value(val[%d], display)' % (fno, vno)
        elif isinstance(f, FormatField):
            formats[f.name] = 'val[%d]' % vno
        else:
            if f.structvalues == 1:
                vrange = str(vno)
            else:
                vrange = '%d:%d' % (vno, vno + f.structvalues)
            if f.parse_value is None:
                code = code + '  ret["%s"] = val[%s]\n' % (f.name, vrange)
            else:
                code = code + '  ret["%s"] = self.static_fields[%d].parse_value(val[%s], display)\n' % (f.name, fno, vrange)
        fno = fno + 1
        vno = vno + f.structvalues
    code = code + '  data = data[%d:]\n' % self.static_size
    fno = 0
    for f in self.var_fields:
        code = code + '  ret["%s"], data = self.var_fields[%d].parse_binary_value(data, display, %s, %s)\n' % (f.name, fno, lengths.get(f.name, 'None'), formats.get(f.name, 'None'))
        fno = fno + 1
    code = code + '  if not rawdict: ret = DictWrapper(ret)\n'
    code = code + '  return ret, data\n'
    g = globals().copy()
    exec(code, g)
    self.parse_binary = types.MethodType(g['parse_binary'], self)
    return self.parse_binary(data, display, rawdict)