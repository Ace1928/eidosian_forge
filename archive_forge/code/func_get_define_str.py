import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def get_define_str(self):
    """construct a define string from the parameters dictionary"""
    define_str_tpl = '\n__title__\na coord\n__inter__\nbb all __basis_set__\n*\neht\ny\n__charge_str____occ_str____single_atom_str____norb_str____dft_str____ri_str____scfiterlimit____fermi_str____damp_str__q\n'
    params = self.parameters
    if params['use redundant internals']:
        internals_str = 'ired\n*'
    else:
        internals_str = '*\nno'
    charge_str = str(params['total charge']) + '\n'
    if params['multiplicity'] == 1:
        if params['uhf']:
            occ_str = 'n\ns\n*\n'
        else:
            occ_str = 'y\n'
    elif params['multiplicity'] == 2:
        occ_str = 'y\n'
    elif params['multiplicity'] == 3:
        occ_str = 'n\nt\n*\n'
    else:
        unpaired = params['multiplicity'] - 1
        if params['use fermi smearing']:
            occ_str = 'n\nuf ' + str(unpaired) + '\n*\n'
        else:
            occ_str = 'n\nu ' + str(unpaired) + '\n*\n'
    if len(self.atoms) != 1:
        single_atom_str = ''
    else:
        single_atom_str = '\n'
    if params['multiplicity'] == 1 and (not params['uhf']):
        norb_str = ''
    else:
        norb_str = 'n\n'
    if params['use dft']:
        dft_str = 'dft\non\n*\n'
    else:
        dft_str = ''
    if params['density functional']:
        dft_str += 'dft\nfunc ' + params['density functional'] + '\n*\n'
    if params['grid size']:
        dft_str += 'dft\ngrid ' + params['grid size'] + '\n*\n'
    if params['use resolution of identity']:
        ri_str = 'ri\non\nm ' + str(params['ri memory']) + '\n*\n'
    else:
        ri_str = ''
    if params['scf iterations']:
        scfmaxiter = params['scf iterations']
        scfiter_str = 'scf\niter\n' + str(scfmaxiter) + '\n\n'
    else:
        scfiter_str = ''
    if params['scf energy convergence']:
        conv = floor(-log10(params['scf energy convergence'] / Ha))
        scfiter_str += 'scf\nconv\n' + str(int(conv)) + '\n\n'
    fermi_str = ''
    if params['use fermi smearing']:
        fermi_str = 'scf\nfermi\n'
        if params['fermi initial temperature']:
            par = str(params['fermi initial temperature'])
            fermi_str += '1\n' + par + '\n'
        if params['fermi final temperature']:
            par = str(params['fermi final temperature'])
            fermi_str += '2\n' + par + '\n'
        if params['fermi annealing factor']:
            par = str(params['fermi annealing factor'])
            fermi_str += '3\n' + par + '\n'
        if params['fermi homo-lumo gap criterion']:
            par = str(params['fermi homo-lumo gap criterion'])
            fermi_str += '4\n' + par + '\n'
        if params['fermi stopping criterion']:
            par = str(params['fermi stopping criterion'])
            fermi_str += '5\n' + par + '\n'
        fermi_str += '\n\n'
    damp_str = ''
    damp_keys = ('initial damping', 'damping adjustment step', 'minimal damping')
    damp_pars = [params[k] for k in damp_keys]
    if any(damp_pars):
        damp_str = 'scf\ndamp\n'
        for par in damp_pars:
            par_str = str(par) if par else ''
            damp_str += par_str + '\n'
        damp_str += '\n'
    define_str = define_str_tpl
    define_str = re.sub('__title__', params['title'], define_str)
    define_str = re.sub('__basis_set__', params['basis set name'], define_str)
    define_str = re.sub('__charge_str__', charge_str, define_str)
    define_str = re.sub('__occ_str__', occ_str, define_str)
    define_str = re.sub('__norb_str__', norb_str, define_str)
    define_str = re.sub('__dft_str__', dft_str, define_str)
    define_str = re.sub('__ri_str__', ri_str, define_str)
    define_str = re.sub('__single_atom_str__', single_atom_str, define_str)
    define_str = re.sub('__inter__', internals_str, define_str)
    define_str = re.sub('__scfiterlimit__', scfiter_str, define_str)
    define_str = re.sub('__fermi_str__', fermi_str, define_str)
    define_str = re.sub('__damp_str__', damp_str, define_str)
    return define_str