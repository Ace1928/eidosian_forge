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
def adjust(self, eldict, block):
    udict = {}
    for elem in self.elems:
        if isinstance(elem, PyAttribute) or isinstance(elem, PyGroup):
            elem.done = True
            continue
        if elem in block:
            continue
        if not elem.done:
            udict[elem] = elem.undefined(eldict)
    keys = [k.name for k in udict.keys()]
    print('#', keys)
    res = (None, [])
    if not udict:
        return res
    level = 1
    rblocked = [p.name for p in block]
    while True:
        non_child = 0
        for objekt, (sup, elems) in udict.items():
            if sup:
                continue
            else:
                non_child += 1
                signif = []
                other = []
                for elem in elems:
                    if elem.name in keys:
                        signif.append(elem)
                    elif elem.ref in rblocked:
                        other.append(elem)
                if len(signif) <= level:
                    alla = signif
                    alla.extend(other)
                    tup = self._mk_list(objekt, alla, eldict)
                    res = (objekt, tup)
                    break
        if res[0]:
            ref = res[0].name
            tups = res[1]
            for objekt, (sups, elems) in udict.items():
                if sups:
                    for sup in sups:
                        if sup.name == ref:
                            for tup in tups:
                                tup[1].append(f'{objekt.class_name}.{tup[2]}')
                            break
                else:
                    pass
        elif not non_child or level > 10:
            elm = udict.keys()[0]
            parent = find_parent(elm, eldict)
            signif = []
            other = []
            tot = parent.properties[0]
            tot.extend(parent.properties[1])
            alla = []
            for elem in tot:
                if isinstance(elem, PyAttribute):
                    continue
                else:
                    alla.append(elem)
            tup = self._mk_list(parent, alla, eldict)
            res = (parent, tup)
        if res[0]:
            break
        else:
            level += 1
    return res