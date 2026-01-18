import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def read_PIC(file: TextIO, verbose: bool=False, quick: bool=False, defaults: bool=False) -> Structure:
    """Load Protein Internal Coordinate (.pic) data from file.

    PIC file format:
        - comment lines start with #
        - (optional) PDB HEADER record
           - idcode and deposition date recommended but optional
           - deposition date in PDB format or as changed by Biopython
        - (optional) PDB TITLE record
        - repeat:
           - Biopython Residue Full ID - sets residue IDs of returned structure
           - (optional) PDB N, CA, C ATOM records for chain start
           - (optional) PIC Hedra records for residue
           - (optional) PIC Dihedra records for residue
           - (optional) BFAC records listing AtomKeys and b-factors

    An improvement would define relative positions for HOH (water) entries.

    Defaults will be supplied for any value if defaults=True.  Default values
    are supplied in ic_data.py, but structures degrade quickly with any
    deviation from true coordinates.  Experiment with
    :data:`Bio.PDB.internal_coords.IC_Residue.pic_flags` options to
    :func:`write_PIC` to verify this.

    N.B. dihedron (i-1)C-N-CA-CB is ignored in assembly if O exists.

    C-beta is by default placed using O-C-CA-CB, but O is missing
    in some PDB file residues, which means the sidechain cannot be
    placed.  The alternate CB path (i-1)C-N-CA-CB is provided to
    circumvent this, but if this is needed then it must be adjusted in
    conjunction with PHI ((i-1)C-N-CA-C) as they overlap (see :meth:`.bond_set`
    and :meth:`.bond_rotate` to handle this automatically).

    :param Bio.File file: :func:`.as_handle` file name or handle
    :param bool verbose: complain when lines not as expected
    :param bool quick: don't check residues for all dihedra (no default values)
    :param bool defaults: create di/hedra as needed from reference database.
        Amide proton created if 'H' is in IC_Residue.accept_atoms
    :returns: Biopython Structure object, Residues with .internal_coord
        attributes but no coordinates except for chain start N, CA, C atoms if
        supplied, **OR** None on parse fail (silent unless verbose=True)

    """
    proton = 'H' in IC_Residue.accept_atoms
    pdb_hdr_re = re.compile('^HEADER\\s{4}(?P<cf>.{1,40})(?:\\s+(?P<dd>\\d\\d\\d\\d-\\d\\d-\\d\\d|\\d\\d-\\w\\w\\w-\\d\\d))?(?:\\s+(?P<id>[0-9A-Z]{4}))?\\s*$')
    pdb_ttl_re = re.compile('^TITLE\\s{5}(?P<ttl>.+)\\s*$')
    biop_id_re = re.compile("^\\('(?P<pid>[^\\s]*)',\\s(?P<mdl>\\d+),\\s'(?P<chn>\\s|\\w)',\\s\\('(?P<het>\\s|[\\w\\s-]+)',\\s(?P<pos>-?\\d+),\\s'(?P<icode>\\s|\\w)'\\)\\)\\s+(?P<res>[\\w]{1,3})(\\s\\[(?P<segid>[a-zA-z\\s]+)\\])?\\s*$")
    pdb_atm_re = re.compile('^ATOM\\s\\s(?:\\s*(?P<ser>\\d+))\\s(?P<atm>[\\w\\s]{4})(?P<alc>\\w|\\s)(?P<res>[\\w]{3})\\s(?P<chn>.)(?P<pos>[\\s\\-\\d]{4})(?P<icode>[A-Za-z\\s])\\s\\s\\s(?P<x>[\\s\\-\\d\\.]{8})(?P<y>[\\s\\-\\d\\.]{8})(?P<z>[\\s\\-\\d\\.]{8})(?P<occ>[\\s\\d\\.]{6})(?P<tfac>[\\s\\d\\.]{6})\\s{6}(?P<segid>[a-zA-z\\s]{4})(?P<elm>.{2})(?P<chg>.{2})?\\s*$')
    pdbx_atm_re = re.compile('^ATOM\\s\\s(?:\\s*(?P<ser>\\d+))\\s(?P<atm>[\\w\\s]{4})(?P<alc>\\w|\\s)(?P<res>[\\w]{3})\\s(?P<chn>.)(?P<pos>[\\s\\-\\d]{4})(?P<icode>[A-Za-z\\s])\\s\\s\\s(?P<x>[\\s\\-\\d\\.]{10})(?P<y>[\\s\\-\\d\\.]{10})(?P<z>[\\s\\-\\d\\.]{10})(?P<occ>[\\s\\d\\.]{7})(?P<tfac>[\\s\\d\\.]{6})\\s{6}(?P<segid>[a-zA-z\\s]{4})(?P<elm>.{2})(?P<chg>.{2})?\\s*$')
    bfac_re = re.compile('^BFAC:\\s([^\\s]+\\s+[\\-\\d\\.]+)\\s*([^\\s]+\\s+[\\-\\d\\.]+)?\\s*([^\\s]+\\s+[\\-\\d\\.]+)?\\s*([^\\s]+\\s+[\\-\\d\\.]+)?\\s*([^\\s]+\\s+[\\-\\d\\.]+)?')
    bfac2_re = re.compile('([^\\s]+)\\s+([\\-\\d\\.]+)')
    struct_builder = StructureBuilder()
    header_dict = _parse_pdb_header_list([])
    curr_SMCS = [None, None, None, None]
    SMCS_init = [struct_builder.init_structure, struct_builder.init_model, struct_builder.init_chain, struct_builder.init_seg]
    sb_res = None
    rkl = None
    sb_chain = None
    sbcic = None
    sbric = None
    akc = {}
    hl12 = {}
    ha = {}
    hl23 = {}
    da = {}
    bfacs = {}
    orphan_aks = set()
    tr = []
    pr = []

    def akcache(akstr: str) -> AtomKey:
        """Maintain dictionary of AtomKeys seen while reading this PIC file."""
        try:
            return akc[akstr]
        except KeyError:
            ak = akc[akstr] = AtomKey(akstr)
            return ak

    def link_residues(ppr: List[Residue], pr: List[Residue]) -> None:
        """Set next and prev links between i-1 and i-2 residues."""
        for p_r in pr:
            pric = p_r.internal_coord
            for p_p_r in ppr:
                ppric = p_p_r.internal_coord
                if p_r.id[0] == ' ':
                    if pric not in ppric.rnext:
                        ppric.rnext.append(pric)
                if p_p_r.id[0] == ' ':
                    if ppric not in pric.rprev:
                        pric.rprev.append(ppric)

    def process_hedron(a1: str, a2: str, a3: str, l12: str, ang: str, l23: str, ric: IC_Residue) -> Tuple:
        """Create Hedron on current (sbcic) Chain.internal_coord."""
        ek = (akcache(a1), akcache(a2), akcache(a3))
        atmNdx = AtomKey.fields.atm
        accpt = IC_Residue.accept_atoms
        if not all((ek[i].akl[atmNdx] in accpt for i in range(3))):
            return
        hl12[ek] = float(l12)
        ha[ek] = float(ang)
        hl23[ek] = float(l23)
        sbcic.hedra[ek] = ric.hedra[ek] = h = Hedron(ek)
        h.cic = sbcic
        ak_add(ek, ric)
        return ek

    def default_hedron(ek: Tuple, ric: IC_Residue) -> None:
        """Create Hedron based on same re_class hedra in ref database.

        Adds Hedron to current Chain.internal_coord, see ic_data for default
        values and reference database source.
        """
        atomkeys = []
        hkey = None
        atmNdx = AtomKey.fields.atm
        resNdx = AtomKey.fields.resname
        resPos = AtomKey.fields.respos
        atomkeys = [ek[i].akl for i in range(3)]
        atpl = tuple([atomkeys[i][atmNdx] for i in range(3)])
        res = atomkeys[0][resNdx]
        if atomkeys[0][resPos] != atomkeys[2][resPos] or atpl == ('N', 'CA', 'C') or atpl in ic_data_backbone or (res not in ['A', 'G'] and atpl in ic_data_sidechains[res]):
            hkey = ek
            rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(3)]
            try:
                dflts = hedra_defaults[''.join(rhcl)][0]
            except KeyError:
                if atomkeys[0][resPos] == atomkeys[1][resPos]:
                    rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(2)]
                    rhc = ''.join(rhcl) + 'X' + atomkeys[2][atmNdx]
                else:
                    rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(1, 3)]
                    rhc = 'X' + atomkeys[0][atmNdx] + ''.join(rhcl)
                dflts = hedra_defaults[rhc][0]
        else:
            hkey = ek[::-1]
            rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(2, -1, -1)]
            dflts = hedra_defaults[''.join(rhcl)][0]
        process_hedron(str(hkey[0]), str(hkey[1]), str(hkey[2]), dflts[0], dflts[1], dflts[2], ric)
        if verbose:
            print(f' default for {ek}')

    def hedra_check(dk: Tuple, ric: IC_Residue) -> None:
        """Confirm both hedra present for dihedron key, use default if set."""
        if dk[0:3] not in sbcic.hedra and dk[2::-1] not in sbcic.hedra:
            if defaults:
                default_hedron(dk[0:3], ric)
            else:
                print(f'{dk} missing h1')
        if dk[1:4] not in sbcic.hedra and dk[3:0:-1] not in sbcic.hedra:
            if defaults:
                default_hedron(dk[1:4], ric)
            else:
                print(f'{dk} missing h2')

    def process_dihedron(a1: str, a2: str, a3: str, a4: str, dangle: str, ric: IC_Residue) -> Set:
        """Create Dihedron on current Chain.internal_coord."""
        ek = (akcache(a1), akcache(a2), akcache(a3), akcache(a4))
        atmNdx = AtomKey.fields.atm
        accpt = IC_Residue.accept_atoms
        if not all((ek[i].akl[atmNdx] in accpt for i in range(4))):
            return
        dangle = float(dangle)
        dangle = dangle if dangle <= 180.0 else dangle - 360.0
        dangle = dangle if dangle >= -180.0 else dangle + 360.0
        da[ek] = float(dangle)
        sbcic.dihedra[ek] = ric.dihedra[ek] = d = Dihedron(ek)
        d.cic = sbcic
        if not quick:
            hedra_check(ek, ric)
        ak_add(ek, ric)
        return ek

    def default_dihedron(ek: List, ric: IC_Residue) -> None:
        """Create Dihedron based on same residue class dihedra in ref database.

        Adds Dihedron to current Chain.internal_coord, see ic_data for default
        values and reference database source.
        """
        atmNdx = AtomKey.fields.atm
        resNdx = AtomKey.fields.resname
        resPos = AtomKey.fields.respos
        rdclass = ''
        dclass = ''
        for ak in ek:
            dclass += ak.akl[atmNdx]
            rdclass += ak.akl[resNdx] + ak.akl[atmNdx]
        if dclass == 'NCACN':
            rdclass = rdclass[0:7] + 'XN'
        elif dclass == 'CACNCA':
            rdclass = 'XCAXC' + rdclass[5:]
        elif dclass == 'CNCAC':
            rdclass = 'XC' + rdclass[2:]
        if rdclass in dihedra_primary_defaults:
            process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), dihedra_primary_defaults[rdclass][0], ric)
            if verbose:
                print(f' default for {ek}')
        elif rdclass in dihedra_secondary_defaults:
            primAngle, offset = dihedra_secondary_defaults[rdclass]
            rname = ek[2].akl[resNdx]
            rnum = int(ek[2].akl[resPos])
            paKey = None
            if primAngle == ('N', 'CA', 'C', 'N') and ek[0].ric.rnext != []:
                paKey = [AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(3)]
                rnext = ek[0].ric.rnext
                paKey.append(AtomKey((rnext[0].rbase[0], None, rnext[0].rbase[2], 'N', None, None)))
                paKey = tuple(paKey)
            elif primAngle == ('CA', 'C', 'N', 'CA'):
                prname = pr.akl[0][resNdx]
                prnum = pr.akl[0][resPos]
                paKey = [AtomKey(prnum, None, prname, primAngle[x], None, None) for x in range(2)]
                paKey.add([AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(2, 4)])
                paKey = tuple(paKey)
            else:
                paKey = tuple((AtomKey((rnum, None, rname, atm, None, None)) for atm in primAngle))
            if paKey in da:
                angl = da[paKey] + dihedra_secondary_defaults[rdclass][1]
                process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), angl, ric)
                if verbose:
                    print(f' secondary default for {ek}')
            elif rdclass in dihedra_secondary_xoxt_defaults:
                if primAngle == ('C', 'N', 'CA', 'C'):
                    prname = pr.akl[0][resNdx]
                    prnum = pr.akl[0][resPos]
                    paKey = [AtomKey(prnum, None, prname, primAngle[0], None, None)]
                    paKey.add([AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(1, 4)])
                    paKey = tuple(paKey)
                else:
                    primAngle, offset = dihedra_secondary_xoxt_defaults[rdclass]
                    rname = ek[2].akl[resNdx]
                    rnum = int(ek[2].akl[resPos])
                    paKey = tuple((AtomKey((rnum, None, rname, atm, None, None)) for atm in primAngle))
                if paKey in da:
                    angl = da[paKey] + offset
                    process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), angl, ric)
                    if verbose:
                        print(f' oxt default for {ek}')
                else:
                    print(f'missing primary angle {paKey} {primAngle} to generate {rnum}{rname} {rdclass}')
        else:
            print(f'missing {ek} -> {rdclass} ({dclass}) not found in primary or secondary defaults')

    def dihedra_check(ric: IC_Residue) -> None:
        """Look for required dihedra in residue, generate defaults if set."""

        def ake_recurse(akList: List) -> List:
            """Bulid combinatorics of AtomKey lists."""
            car = akList[0]
            if len(akList) > 1:
                retList = []
                for ak in car:
                    cdr = akList[1:]
                    rslt = ake_recurse(cdr)
                    for r in rslt:
                        r.insert(0, ak)
                        retList.append(r)
                return retList
            elif len(car) == 1:
                return [list(car)]
            else:
                retList = [[ak] for ak in car]
                return retList

        def ak_expand(eLst: List) -> List:
            """Expand AtomKey list with altlocs, all combinatorics."""
            retList = []
            for edron in eLst:
                newList = []
                for ak in edron:
                    rslt = ak.ric.split_akl([ak])
                    rlst = [r[0] for r in rslt]
                    if rlst != []:
                        newList.append(rlst)
                    else:
                        newList.append([ak])
                rslt = ake_recurse(newList)
                for r in rslt:
                    retList.append(r)
            return retList
        chkLst = []
        sN, sCA, sC = (AtomKey(ric, 'N'), AtomKey(ric, 'CA'), AtomKey(ric, 'C'))
        sO, sCB, sH = (AtomKey(ric, 'O'), AtomKey(ric, 'CB'), AtomKey(ric, 'H'))
        if ric.rnext != []:
            for rn in ric.rnext:
                nN, nCA, nC = (AtomKey(rn, 'N'), AtomKey(rn, 'CA'), AtomKey(rn, 'C'))
                chkLst.append((sN, sCA, sC, nN))
                chkLst.append((sCA, sC, nN, nCA))
                chkLst.append((sC, nN, nCA, nC))
        else:
            chkLst.append((sN, sCA, sC, AtomKey(ric, 'OXT')))
            rn = '(no rnext)'
        chkLst.append((sN, sCA, sC, sO))
        if ric.lc != 'G':
            chkLst.append((sO, sC, sCA, sCB))
            if ric.lc == 'A':
                chkLst.append((sN, sCA, sCB))
        if ric.rprev != [] and ric.lc != 'P' and proton:
            chkLst.append((sC, sCA, sN, sH))
        try:
            for edron in ic_data_sidechains[ric.lc]:
                if len(edron) > 3:
                    if all((atm[0] != 'H' for atm in edron)):
                        akl = [AtomKey(ric, atm) for atm in edron[0:4]]
                        chkLst.append(akl)
        except KeyError:
            pass
        chkLst = ak_expand(chkLst)
        altloc_ndx = AtomKey.fields.altloc
        for dk in chkLst:
            if tuple(dk) in ric.dihedra:
                pass
            elif sH in dk:
                pass
            elif all((atm.akl[altloc_ndx] is None for atm in dk)):
                if defaults:
                    if len(dk) != 3:
                        default_dihedron(dk, ric)
                    else:
                        default_hedron(dk, ric)
                elif verbose:
                    print(f'{ric}-{rn} missing {dk}')
            else:
                pass

    def ak_add(ek: Tuple, ric: IC_Residue) -> None:
        """Allocate edron key AtomKeys to current residue as appropriate.

        A hedron or dihedron may span a backbone amide bond, this routine
        allocates atoms in the (h/di)edron to the ric residue or saves them
        for a residue yet to be processed.

        :param set ek: AtomKeys in edron
        :param IC_Residue ric: current residue to assign AtomKeys to
        """
        res = ric.residue
        reskl = (str(res.id[1]), None if res.id[2] == ' ' else res.id[2], ric.lc)
        for ak in ek:
            if ak.ric is None:
                sbcic.akset.add(ak)
                if ak.akl[0:3] == reskl:
                    ak.ric = ric
                    ric.ak_set.add(ak)
                else:
                    orphan_aks.add(ak)

    def finish_chain() -> None:
        """Do last rnext, rprev links and process chain edra data."""
        link_residues(pr, tr)
        if not quick:
            for r in pr:
                dihedra_check(r.internal_coord)
            for r in tr:
                dihedra_check(r.internal_coord)
        if ha != {}:
            sha = {k: ha[k] for k in sorted(ha)}
            shl12 = {k: hl12[k] for k in sorted(hl12)}
            shl23 = {k: hl23[k] for k in sorted(hl23)}
            sda = {k: da[k] for k in sorted(da)}
            sbcic._hedraDict2chain(shl12, sha, shl23, sda, bfacs)
    with as_handle(file, mode='r') as handle:
        for line in handle.readlines():
            if line.startswith('#'):
                pass
            elif line.startswith('HEADER '):
                m = pdb_hdr_re.match(line)
                if m:
                    header_dict['head'] = m.group('cf')
                    header_dict['idcode'] = m.group('id')
                    header_dict['deposition_date'] = m.group('dd')
                elif verbose:
                    print('Reading pic file', file, 'HEADER parse fail: ', line)
            elif line.startswith('TITLE '):
                m = pdb_ttl_re.match(line)
                if m:
                    header_dict['name'] = m.group('ttl').strip()
                elif verbose:
                    print('Reading pic file', file, 'TITLE parse fail:, ', line)
            elif line.startswith('('):
                m = biop_id_re.match(line)
                if m:
                    segid = m.group(9)
                    if segid is None:
                        segid = '    '
                    this_SMCS = [m.group(1), int(m.group(2)), m.group(3), segid]
                    if curr_SMCS != this_SMCS:
                        if curr_SMCS[:3] != this_SMCS[:3] and ha != {}:
                            finish_chain()
                            akc = {}
                            hl12 = {}
                            ha = {}
                            hl23 = {}
                            da = {}
                            bfacs = {}
                        for i in range(4):
                            if curr_SMCS[i] != this_SMCS[i]:
                                SMCS_init[i](this_SMCS[i])
                                curr_SMCS[i] = this_SMCS[i]
                                if i == 0:
                                    struct_builder.set_header(header_dict)
                                elif i == 1:
                                    curr_SMCS[2] = curr_SMCS[3] = None
                                elif i == 2:
                                    sb_chain = struct_builder.chain
                                    sbcic = sb_chain.internal_coord = IC_Chain(sb_chain)
                    struct_builder.init_residue(m.group('res'), m.group('het'), int(m.group('pos')), m.group('icode'))
                    sb_res = struct_builder.residue
                    if sb_res.id[0] != ' ':
                        continue
                    if 2 == sb_res.is_disordered():
                        for r in sb_res.child_dict.values():
                            if not r.internal_coord:
                                sb_res = r
                                break
                        tr.append(sb_res)
                    else:
                        link_residues(pr, tr)
                        if not quick:
                            for r in pr:
                                dihedra_check(r.internal_coord)
                        pr = tr
                        tr = [sb_res]
                    sbric = sb_res.internal_coord = IC_Residue(sb_res)
                    sbric.cic = sbcic
                    rkl = (str(sb_res.id[1]), None if sb_res.id[2] == ' ' else sb_res.id[2], sbric.lc)
                    sbcic.ordered_aa_ic_list.append(sbric)
                    for ak in orphan_aks:
                        if ak.akl[0:3] == rkl:
                            ak.ric = sbric
                            sbric.ak_set.add(ak)
                    orphan_aks = set(filter(lambda ak: ak.ric is None, orphan_aks))
                else:
                    if verbose:
                        print('Reading pic file', file, 'residue ID parse fail: ', line)
                    return None
            elif line.startswith('ATOM '):
                m = pdb_atm_re.match(line)
                if not m:
                    m = pdbx_atm_re.match(line)
                if m:
                    if sb_res is None:
                        if verbose:
                            print('Reading pic file', file, 'ATOM without residue configured:, ', line)
                        return None
                    if sb_res.resname != m.group('res') or sb_res.id[1] != int(m.group('pos')):
                        if verbose:
                            print('Reading pic file', file, 'ATOM not in configured residue (', sb_res.resname, str(sb_res.id), '):', line)
                        return None
                    coord = np.array((float(m.group('x')), float(m.group('y')), float(m.group('z'))), 'f')
                    struct_builder.init_atom(m.group('atm').strip(), coord, float(m.group('tfac')), float(m.group('occ')), m.group('alc'), m.group('atm'), int(m.group('ser')), m.group('elm').strip())
                    pr = []
            elif line.startswith('BFAC: '):
                m = bfac_re.match(line)
                if m:
                    for bfac_pair in m.groups():
                        if bfac_pair is not None:
                            m2 = bfac2_re.match(bfac_pair)
                            bfacs[m2.group(1)] = float(m2.group(2))
            else:
                m = Edron.edron_re.match(line)
                if m and sb_res is not None:
                    if m['a4'] is None:
                        process_hedron(m['a1'], m['a2'], m['a3'], m['len12'], m['angle'], m['len23'], sb_res.internal_coord)
                    else:
                        process_dihedron(m['a1'], m['a2'], m['a3'], m['a4'], m['dihedral'], sb_res.internal_coord)
                elif m:
                    print('PIC file: ', file, ' error: no residue info before reading (di/h)edron: ', line)
                    return None
                elif line.strip():
                    if verbose:
                        print('Reading PIC file', file, 'parse fail on: .', line, '.')
                    return None
    finish_chain()
    return struct_builder.get_structure()