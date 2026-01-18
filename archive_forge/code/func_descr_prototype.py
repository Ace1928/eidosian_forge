import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
def descr_prototype(self, buf):
    """
        Describe the prototype ("head") of the function.
        """
    state = 'define' if self.blocks else 'declare'
    ret = self.return_value
    args = ', '.join((str(a) for a in self.args))
    name = self.get_reference()
    attrs = ' ' + ' '.join(self.attributes._to_list(self.ftype.return_type)) if self.attributes else ''
    if any(self.args):
        vararg = ', ...' if self.ftype.var_arg else ''
    else:
        vararg = '...' if self.ftype.var_arg else ''
    linkage = self.linkage
    cconv = self.calling_convention
    prefix = ' '.join((str(x) for x in [state, linkage, cconv, ret] if x))
    metadata = self._stringify_metadata()
    metadata = ' {}'.format(metadata) if metadata else ''
    section = ' section "{}"'.format(self.section) if self.section else ''
    pt_str = '{prefix} {name}({args}{vararg}){attrs}{section}{metadata}\n'
    prototype = pt_str.format(prefix=prefix, name=name, args=args, vararg=vararg, attrs=attrs, section=section, metadata=metadata)
    buf.append(prototype)