import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def distance_to_internal_coordinates(self, resetAtoms: Optional[Union[bool, None]]=True) -> None:
    """Compute chain di/hedra from from distance and chirality data.

        Distance properties on hedra L12, L23, L13 and dihedra L14 configured
        by :meth:`.distplot_to_dh_arrays` or alternative loader.

        dihedraAngles result is multiplied by dihedra_signs at final step
        recover chirality information lost in distance plot (mirror image of
        structure has same distances but opposite sign dihedral angles).

        Note that chain breaks will cause errors in rebuilt structure, use
        :meth:`.copy_initNCaCs` to avoid this

        Based on Blue, the Hedronometer's answer to `The dihedral angles of a tetrahedron
        in terms of its edge lengths <https://math.stackexchange.com/a/49340/972353>`_
        on `math.stackexchange.com <https://math.stackexchange.com/>`_.  See also:
        `"Heron-like Hedronometric Results for Tetrahedral Volume"
        <http://daylateanddollarshort.com/mathdocs/Heron-like-Results-for-Tetrahedral-Volume.pdf>`_.

        Other values from that analysis included here as comments for
        completeness:

        * oa = hedron1 L12 if reverse else hedron1 L23
        * ob = hedron1 L23 if reverse else hedron1 L12
        * ac = hedron2 L12 if reverse else hedron2 L23
        * ab = hedron1 L13 = law of cosines on OA, OB (hedron1 L12, L23)
        * oc = hedron2 L13 = law of cosines on OA, AC (hedron2 L12, L23)
        * bc = dihedron L14

        target is OA, the dihedral angle along edge oa.

        :param bool resetAtoms: default True.
            Mark all atoms in di/hedra and atomArray for updating by
            :meth:`.internal_to_atom_coordinates`.  Alternatvely set this to
            False and manipulate `atomArrayValid`, `dAtoms_needs_update` and
            `hAtoms_needs_update` directly to reduce computation.
        """
    oa = self.hedraL12[self.dH1ndx]
    oa[self.dFwd] = self.hedraL23[self.dH1ndx][self.dFwd]
    ob = self.hedraL23[self.dH1ndx]
    ob[self.dFwd] = self.hedraL12[self.dH1ndx][self.dFwd]
    ac = self.hedraL12[self.dH2ndx]
    ac[self.dFwd] = self.hedraL23[self.dH2ndx][self.dFwd]
    ab = self.hedraL13[self.dH1ndx]
    oc = self.hedraL13[self.dH2ndx]
    bc = self.dihedraL14
    Ys = (oa + ac + oc) / 2
    Zs = (oa + ob + ab) / 2
    Ysqr = Ys * (Ys - oa) * (Ys - ac) * (Ys - oc)
    Zsqr = Zs * (Zs - oa) * (Zs - ob) * (Zs - ab)
    Hsqr = (4 * oa * oa * bc * bc - np.square(ob * ob + ac * ac - (oc * oc + ab * ab))) / 16
    '\n        Jsqr = (\n            4 * ob * ob * ac * ac\n            - np.square((oc * oc + ab * ab) - (oa * oa + bc * bc))\n        ) / 16\n        Ksqr = (\n            4 * oc * oc * ab * ab\n            - np.square((oa * oa + bc * bc) - (ob * ob + ac * ac))\n        ) / 16\n        '
    Y = np.sqrt(Ysqr)
    Z = np.sqrt(Zsqr)
    cosOA = (Ysqr + Zsqr - Hsqr) / (2 * Y * Z)
    cosOA[cosOA < -1.0] = -1.0
    cosOA[cosOA > 1.0] = 1.0
    np.arccos(cosOA, out=self.dihedraAngleRads, dtype=np.longdouble)
    self.dihedraAngleRads *= self.dihedra_signs
    np.rad2deg(self.dihedraAngleRads, out=self.dihedraAngle)
    np.rad2deg(np.arccos((np.square(self.hedraL12) + np.square(self.hedraL23) - np.square(self.hedraL13)) / (2 * self.hedraL12 * self.hedraL23)), out=self.hedraAngle)
    if resetAtoms:
        self.atomArrayValid[:] = False
        self.dAtoms_needs_update[:] = True
        self.hAtoms_needs_update[:] = True