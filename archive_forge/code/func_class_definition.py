import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
def class_definition(self, target_namespace, cdict=None, ignore=None):
    line = []
    if self.root:
        if self.name not in [c.name for c in self.root.elems]:
            self.root.elems.append(self)
    superior, sups, imps = self._superiors(cdict)
    c_name = klass_namn(self)
    if not superior:
        line.append(f'class {c_name}(SamlBase):')
    else:
        line.append(f'class {c_name}({','.join(sups)}):')
    if hasattr(self, 'scoped'):
        pass
    else:
        line.append(f'{INDENT}"""The {target_namespace}:{self.name} element """')
    line.append('')
    line.append(f"{INDENT}c_tag = '{self.name}'")
    line.append(f'{INDENT}c_namespace = NAMESPACE')
    try:
        if self.value_type:
            if isinstance(self.value_type, str):
                line.append(f"{INDENT}c_value_type = '{self.value_type}'")
            else:
                line.append(f'{INDENT}c_value_type = {self.value_type}')
    except AttributeError:
        pass
    if not superior:
        for var, cps in CLASS_PROP:
            line.append(f'{INDENT}{var} = SamlBase.{var}{cps}')
    else:
        for sup in sups:
            for var, cps in CLASS_PROP:
                line.append(f'{INDENT}{var} = {sup}.{var}{cps}')
    args, child, inh = self._do_properties(line, cdict, ignore, target_namespace)
    if child:
        line.append('{}c_child_order.extend([{}])'.format(INDENT, "'" + "', '".join(child) + "'"))
    if args:
        if inh:
            cname = self.knamn(self.superior[0], cdict)
            imps = {cname: [c.pyname for c in inh if c.pyname]}
        line.append('')
        line.extend(def_init(imps, args))
        line.extend(base_init(imps))
        line.extend(initialize(args))
    line.append('')
    if not self.abstract or not self.class_name.endswith('_'):
        line.append(f'def {pyify(self.class_name)}_from_string(xml_string):')
        line.append(f'{INDENT}return saml2.create_class_from_xml_string({self.class_name}, xml_string)')
        line.append('')
    self.done = True
    return '\n'.join(line)