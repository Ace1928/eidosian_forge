import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def get_ppp_list(atoms, species, raise_exception, xc, pps, search_paths):
    ppp_list = []
    xcname = 'GGA' if xc != 'LDA' else 'LDA'
    for Z in species:
        number = abs(Z)
        symbol = chemical_symbols[number]
        names = []
        for s in [symbol, symbol.lower()]:
            for xcn in [xcname, xcname.lower()]:
                if pps in ['paw']:
                    hghtemplate = '%s-%s-%s.paw'
                    names.append(hghtemplate % (s, xcn, '*'))
                    names.append('%s[.-_]*.paw' % s)
                elif pps in ['pawxml']:
                    hghtemplate = '%s.%s%s.xml'
                    names.append(hghtemplate % (s, xcn, '*'))
                    names.append('%s[.-_]*.xml' % s)
                elif pps in ['hgh.k']:
                    hghtemplate = '%s-q%s.hgh.k'
                    names.append(hghtemplate % (s, '*'))
                    names.append('%s[.-_]*.hgh.k' % s)
                    names.append('%s[.-_]*.hgh' % s)
                elif pps in ['tm']:
                    hghtemplate = '%d%s%s.pspnc'
                    names.append(hghtemplate % (number, s, '*'))
                    names.append('%s[.-_]*.pspnc' % s)
                elif pps in ['hgh', 'hgh.sc']:
                    hghtemplate = '%d%s.%s.hgh'
                    names.append(hghtemplate % (number, s, '*'))
                    names.append('%d%s%s.hgh' % (number, s, '*'))
                    names.append('%s[.-_]*.hgh' % s)
                else:
                    names.append('%02d-%s.%s.%s' % (number, s, xcn, pps))
                    names.append('%02d[.-_]%s*.%s' % (number, s, pps))
                    names.append('%02d%s*.%s' % (number, s, pps))
                    names.append('%s[.-_]*.%s' % (s, pps))
        found = False
        for name in names:
            for path in search_paths:
                filenames = glob(join(path, name))
                if not filenames:
                    continue
                if pps == 'paw':
                    filenames[0] = max(filenames)
                elif pps == 'hgh':
                    filenames[0] = min(filenames)
                elif pps == 'hgh.k':
                    filenames[0] = max(filenames)
                elif pps == 'tm':
                    filenames[0] = max(filenames)
                elif pps == 'hgh.sc':
                    filenames[0] = max(filenames)
                if filenames:
                    found = True
                    ppp_list.append(filenames[0])
                    break
            if found:
                break
        if not found:
            ppp_list.append('Provide {}.{}.{}?'.format(symbol, '*', pps))
            if raise_exception:
                msg = 'Could not find {} pseudopotential {} for {}'.format(xcname.lower(), pps, symbol)
                raise RuntimeError(msg)
    return ppp_list